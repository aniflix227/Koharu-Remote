use anyhow::{Result, bail, anyhow};
use async_trait::async_trait;
use koharu_core::{ImageRole, MaskRole, Op, TextData, NodeId};
use serde::{Deserialize, Serialize};
use reqwest_middleware::reqwest::{Client, multipart};
use std::io::Cursor;

use crate::pipeline::artifacts::Artifact;
use crate::pipeline::engine::{Engine, EngineCtx, EngineInfo};
use crate::pipeline::engines::support::{
    clear_text_nodes_ops, load_source_image, new_text_node, page_node_count,
    sort_manga_reading_order, text_region_to_pair, find_mask_node, upsert_image_blob, image_dimensions
};

#[derive(Debug, Deserialize)]
struct RemoteTextRegion {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    detector: Option<String>,
}

impl Into<koharu_ml::types::TextRegion> for RemoteTextRegion {
    fn into(self) -> koharu_ml::types::TextRegion {
        koharu_ml::types::TextRegion {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            confidence: 1.0,
            line_polygons: None,
            source_direction: None,
            rotation_deg: None,
            detected_font_size_px: None,
            detector: self.detector,
        }
    }
}

// ---------------------------------------------------------------------------
// Remote Detector
// ---------------------------------------------------------------------------
pub struct RemoteDetectorModel;

#[async_trait]
impl Engine for RemoteDetectorModel {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let remote_url = match &ctx.options.remote_ai_url {
            Some(url) if !url.is_empty() => url,
            _ => bail!("Remote AI URL is not configured. Please set it in Settings."),
        };

        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;
        let mut buf = Cursor::new(Vec::new());
        image.write_to(&mut buf, image::ImageFormat::Jpeg)?;
        let image_bytes = buf.into_inner();

        let client = Client::new();
        let part = multipart::Part::bytes(image_bytes)
            .file_name("image.jpg")
            .mime_str("image/jpeg")?;
        let form = multipart::Form::new().part("image", part);

        let url = format!("{}/v1/detect", remote_url.trim_end_matches('/'));
        let res = client.post(&url).multipart(form).send().await?;
        
        if !res.status().is_success() {
            bail!("Remote detector failed: {}", res.status());
        }

        let regions: Vec<RemoteTextRegion> = res.json().await?;

        let mut ops = clear_text_nodes_ops(ctx.scene, ctx.page);
        let removed = ops.len();
        let mut running_len = page_node_count(ctx.scene, ctx.page).saturating_sub(removed);

        let mut pairs: Vec<([f32; 4], TextData)> = regions
            .into_iter()
            .map(|r| text_region_to_pair(r.into(), "remote-detector"))
            .collect();
            
        sort_manga_reading_order(&mut pairs, ctx.options.reading_order.unwrap_or_default());
        
        for (bbox, text) in pairs {
            let node = new_text_node(bbox, text);
            ops.push(Op::AddNode {
                page: ctx.page,
                node,
                at: running_len,
            });
            running_len += 1;
        }
        
        Ok(ops)
    }
}

inventory::submit! {
    EngineInfo {
        id: "remote-detector",
        name: "Remote AI (Detector)",
        needs: &[],
        produces: &[Artifact::TextBoxes],
        load: |_runtime, _cpu| Box::pin(async move {
            Ok(Box::new(RemoteDetectorModel) as Box<dyn Engine>)
        }),
    }
}

// ---------------------------------------------------------------------------
// Remote OCR
// ---------------------------------------------------------------------------
pub struct RemoteOcrModel;

#[async_trait]
impl Engine for RemoteOcrModel {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let remote_url = match &ctx.options.remote_ai_url {
            Some(url) if !url.is_empty() => url,
            _ => bail!("Remote AI URL is not configured. Please set it in Settings."),
        };

        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;
        let mut ops = Vec::new();
        
        let nodes: Vec<NodeId> = match &ctx.options.text_node_ids {
            Some(ids) => ids.clone(),
            None => crate::pipeline::engines::support::text_nodes(ctx.scene, ctx.page).into_iter().map(|(id, _, _)| id).collect(),
        };

        let client = Client::new();
        let url = format!("{}/v1/ocr", remote_url.trim_end_matches('/'));

        for node_id in nodes {
            let page = match ctx.scene.page(ctx.page) {
                Some(p) => p,
                None => continue,
            };
            let node = match page.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };
            let text_data = match &node.kind {
                koharu_core::NodeKind::Text(t) => t,
                _ => continue,
            };
            
            let bbox = (
                node.transform.x,
                node.transform.y,
                node.transform.x + node.transform.width,
                node.transform.y + node.transform.height,
            );

            let region = koharu_ml::comic_text_detector::crop_text_block_bbox(&image, &koharu_ml::types::TextRegion {
                x: bbox.0,
                y: bbox.1,
                width: bbox.2 - bbox.0,
                height: bbox.3 - bbox.1,
                confidence: 1.0,
                line_polygons: None,
                source_direction: None,
                rotation_deg: None,
                detected_font_size_px: None,
                detector: None,
            });
            let mut buf = Cursor::new(Vec::new());
            region.write_to(&mut buf, image::ImageFormat::Jpeg)?;
            let image_bytes = buf.into_inner();

            let part = multipart::Part::bytes(image_bytes)
                .file_name("crop.jpg")
                .mime_str("image/jpeg")?;
            let form = multipart::Form::new().part("image", part);

            let res = client.post(&url).multipart(form).send().await?;
            if !res.status().is_success() {
                tracing::warn!("Remote OCR failed for node {}: {}", node_id.0, res.status());
                continue;
            }

            #[derive(Deserialize)]
            struct OcrResponse {
                text: String,
            }
            
            let ocr_res: OcrResponse = match res.json().await {
                Ok(data) => data,
                Err(e) => {
                    tracing::warn!("Failed to parse OCR response: {}", e);
                    continue;
                }
            };

            if ocr_res.text.is_empty() {
                continue;
            }
            
            ops.push(Op::UpdateNode {
                page: ctx.page,
                id: node_id,
                patch: koharu_core::NodePatch {
                    data: Some(koharu_core::NodeDataPatch::Text(koharu_core::TextDataPatch {
                        text: Some(Some(ocr_res.text)),
                        ..Default::default()
                    })),
                    transform: None,
                    visible: None,
                },
                prev: koharu_core::NodePatch::default(),
            });
        }

        Ok(ops)
    }
}

inventory::submit! {
    EngineInfo {
        id: "remote-ocr",
        name: "Remote AI (OCR)",
        needs: &[Artifact::TextBoxes],
        produces: &[Artifact::OcrText],
        load: |_runtime, _cpu| Box::pin(async move {
            Ok(Box::new(RemoteOcrModel) as Box<dyn Engine>)
        }),
    }
}

// ---------------------------------------------------------------------------
// Remote Inpainter
// ---------------------------------------------------------------------------
pub struct RemoteInpainterModel;

#[async_trait]
impl Engine for RemoteInpainterModel {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let remote_url = match &ctx.options.remote_ai_url {
            Some(url) if !url.is_empty() => url,
            _ => bail!("Remote AI URL is not configured. Please set it in Settings."),
        };

        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;
        let (_, mask_ref) = find_mask_node(ctx.scene, ctx.page, MaskRole::Segment)
            .ok_or_else(|| anyhow!("no Segment mask on page"))?;
        let mask = ctx.blobs.load_image(&mask_ref)?;

        let mut img_buf = Cursor::new(Vec::new());
        image.write_to(&mut img_buf, image::ImageFormat::Jpeg)?;
        
        let mut mask_buf = Cursor::new(Vec::new());
        mask.write_to(&mut mask_buf, image::ImageFormat::Png)?;

        let client = Client::new();
        let url = format!("{}/v1/inpaint", remote_url.trim_end_matches('/'));

        let form = multipart::Form::new()
            .part("image", multipart::Part::bytes(img_buf.into_inner()).file_name("image.jpg").mime_str("image/jpeg")?)
            .part("mask", multipart::Part::bytes(mask_buf.into_inner()).file_name("mask.png").mime_str("image/png")?);

        let res = client.post(&url).multipart(form).send().await?;
        if !res.status().is_success() {
            bail!("Remote inpainter failed: {}", res.status());
        }

        let bytes = res.bytes().await?.to_vec();
        let result_img = image::load_from_memory(&bytes)?;

        let (w, h) = image_dimensions(&result_img);
        let blob = ctx.blobs.put_webp(&result_img)?;
        
        Ok(vec![upsert_image_blob(
            ctx.scene,
            ctx.page,
            ImageRole::Inpainted,
            blob,
            w,
            h,
        )])
    }
}

inventory::submit! {
    EngineInfo {
        id: "remote-inpainter",
        name: "Remote AI (Inpainter)",
        needs: &[Artifact::SegmentMask],
        produces: &[Artifact::Inpainted],
        load: |_runtime, _cpu| Box::pin(async move {
            Ok(Box::new(RemoteInpainterModel) as Box<dyn Engine>)
        }),
    }
}
