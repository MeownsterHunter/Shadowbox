use super::*;
use crate::ue_shaderlib as ue;

pub(super) struct UeStage {
    pub meta: Value,
    pub stage: headers::Stage,
}

pub(super) fn label(source: &str, meta: &Value) -> String {
    let inner = meta["pak_entry"]
        .as_str()
        .or_else(|| meta["iostore"]["chunk_id"].as_str())
        .map(|name| format!("::{name}"))
        .unwrap_or_default();
    let shader = meta["shader"]["shader_index"]
        .as_u64()
        .map(|index| format!("::shader[{index}]"))
        .unwrap_or_default();
    format!("{source}{inner}{shader}")
}

impl Scanner<'_> {
    pub(super) fn ue_library(
        &mut self,
        source: &str,
        source_bytes: Value,
        context: Value,
        path: &Path,
    ) -> io::Result<()> {
        let parsed = ue::read(&mut File::open(path)?)
            .and_then(|data| ue::loose(&data).map(|library| (data, library)));
        let (data, library) = match parsed {
            Ok(v) => v,
            Err(error) => {
                let reason = format!("library not parsed: {error}");
                writeln!(
                    self.report,
                    "{}: raw-scanned ({reason})",
                    label(source, &context)
                )?;
                self.event(json!({"kind": "ue_library", "source_file": source,
                    "pak_entry": context["pak_entry"], "integrity": context["integrity"],
                    "unreal": context, "status": "raw-scanned", "reason": reason}))?;
                // Retain the original input path and
                // raw validator, including records past the parser's size limit.
                return self.records(
                    path,
                    source,
                    source_bytes,
                    RecordSource::UnrealRaw(&context),
                );
            }
        };
        let mut meta = context.clone();
        meta["library"] = library.metadata;
        meta["shaders"] = json!(library.shaders.len());
        self.ue_event(source, meta, "ue_library", "parsed", None)?;
        for shader in &library.shaders {
            let mut meta = context.clone();
            meta["shader"] = shader.metadata();
            let start = library.code_start + shader.offset as usize;
            let stored = &data[start..start + shader.size as usize];
            match ue::expand(stored, shader.size, shader.uncompressed) {
                Ok((blob, method)) => {
                    meta["compression"] = json!(method);
                    self.ue_blob(
                        source,
                        source_bytes.clone(),
                        meta,
                        shader.stage().map_err(|e| e.to_string()),
                        &blob,
                    )?;
                }
                Err(error) => {
                    // Failed expansion still scans this shader's exact
                    // stored extent. Expanded offsets never bound stored bytes.
                    meta["compression"] = json!("undecoded");
                    self.ue_blob(
                        source,
                        source_bytes.clone(),
                        meta,
                        Err(format!("shader decompression failed: {error}")),
                        stored,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn ue_blob(
        &mut self,
        source: &str,
        source_bytes: Value,
        mut meta: Value,
        stage: Result<headers::Stage, String>,
        blob: &[u8],
    ) -> io::Result<()> {
        // Keep the exact platform payload, including opaque wrapper/tail bytes.
        // Program discovery cannot establish their meaning or discard them.
        let blob_hash = format!("{:x}", Sha256::digest(blob));
        let blob_output = format!("{blob_hash}.ue-shader.bin");
        write_asset(&self.output.join(&blob_output), blob)?;
        meta["blob_sha256"] = json!(blob_hash);
        meta["blob_output"] = json!(blob_output);
        meta["blob_size"] = json!(blob.len());
        let (context, fallback_reason) = match stage {
            Ok(stage) => {
                meta["stage"] = json!(stage_name(stage));
                (
                    Some(UeStage {
                        meta: meta.clone(),
                        stage,
                    }),
                    None,
                )
            }
            Err(reason) => {
                // The strict raw validator determines the record stage
                // when library metadata or compression cannot establish it.
                meta["raw_scan_reason"] = json!(reason);
                (None, Some(reason))
            }
        };
        let before = (
            self.totals.candidates,
            self.totals.accepted,
            self.totals.rejected,
        );
        let (pending, mut file) = PendingFile::new(&self.output.join("ue-shader"))?;
        file.write_all(blob)?;
        drop(file);
        self.records(
            &pending.temporary,
            source,
            source_bytes,
            match &context {
                Some(context) => RecordSource::Unreal(context),
                None => RecordSource::UnrealRaw(&meta),
            },
        )?;
        let (candidates, accepted, rejected) = (
            self.totals.candidates - before.0,
            self.totals.accepted - before.1,
            self.totals.rejected - before.2,
        );
        meta["candidates"] = json!(candidates);
        meta["accepted"] = json!(accepted);
        meta["rejected"] = json!(rejected);
        let kind = if meta["shader"].is_null() {
            "ue_group"
        } else {
            "ue_shader"
        };
        self.ue_event(
            source,
            meta,
            kind,
            if fallback_reason.is_some() {
                "raw-scanned"
            } else if accepted > 0 && rejected == 0 {
                "scanned"
            } else {
                "failed"
            },
            if candidates == 0 {
                Some(format!(
                    "{}unsupported platform blob: no known Maxwell program record",
                    fallback_reason
                        .map(|r| format!("{r}; "))
                        .unwrap_or_default()
                ))
            } else {
                fallback_reason
            },
        )
    }

    pub(super) fn ue_event(
        &mut self,
        source: &str,
        meta: Value,
        kind: &str,
        status: &str,
        reason: Option<String>,
    ) -> io::Result<()> {
        let label = label(source, &meta);
        let stage = meta["stage"]
            .as_str()
            .map(|s| format!(" stage={s}"))
            .unwrap_or_default();
        writeln!(
            self.report,
            "{label}{stage}: {status}{}",
            reason
                .as_ref()
                .map(|s| format!(": {s}"))
                .unwrap_or_default()
        )?;
        // Project reporting policy: docs/reference-sources.md.
        // Library parsing and expansion cannot upgrade the pak checksum evidence.
        self.event(json!({"kind": kind, "source_file": source,
            "pak_entry": meta["pak_entry"], "integrity": meta["integrity"],
            "unreal": meta, "status": status, "reason": reason}))
    }
}
