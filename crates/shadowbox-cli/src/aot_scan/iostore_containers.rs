use super::*;
use crate::iostore::{self, shaders, Toc};

impl Scanner<'_> {
    pub(super) fn iostore(
        &mut self,
        path: &Path,
        source: &str,
        source_bytes: Value,
        toc: Toc,
    ) -> io::Result<()> {
        self.event(json!({"kind": "iostore", "source_file": source, "source_path_hex": source_bytes,
            "version": toc.version, "container_id": toc.container_id, "encryption_guid": toc.encryption_guid,
            "flags": toc.flags, "header_hex": iostore::hex(&toc.header), "mount_point": toc.mount,
            "entries": toc.chunks.len(), "compression_methods": toc.methods, "compression_block_size": toc.block_size,
            "partition_count": toc.partition_count, "partition_size": toc.partition_size,
            "signature_verification": "not_performed", "chunk_hash_verification": "not_performed"}))?;
        // Library headers precede group scanning even when TOC order is different.
        // Only descriptors in this container are resolved. Missing cross-container
        // descriptors remain explicit limitations; no group sizes are guessed.
        let mut groups = HashMap::<[u8; 12], (shaders::Group, Value)>::new();
        let mut conflicts = HashSet::new();
        for (index, chunk) in toc
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| toc.shader_library(c))
        {
            let mut meta = chunk_meta(source, &source_bytes, &toc, index);
            let result = (|| {
                let mut bytes = Vec::new();
                toc.extract(path, chunk, self.pak_key.as_ref(), &mut bytes)?;
                shaders::library(&bytes)
            })();
            match result {
                Ok(library) => {
                    meta["groups"] = json!(library.groups.len());
                    meta["library"] = library.metadata;
                    for group in library.groups {
                        if groups
                            .insert(group.id, (group.clone(), meta.clone()))
                            .is_some_and(|previous| previous.0 != group)
                        {
                            conflicts.insert(group.id);
                        }
                    }
                    self.iostore_event(meta, "library_parsed", None)?;
                }
                Err(error) => self.iostore_event(meta, "skipped", Some(error.to_string()))?,
            }
        }
        for (index, chunk) in toc
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| !toc.shader_library(c))
        {
            let mut meta = chunk_meta(source, &source_bytes, &toc, index);
            let named = chunk.name.as_ref().is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.ends_with(".ushaderbytecode")
                    || name.ends_with(".uzenshaderbytecode")
                    || name.contains("shaderarchive")
                    || name.contains("shadercodelibrary")
            });
            let group = groups.get(&chunk.id);
            if conflicts.contains(&chunk.id) {
                self.iostore_event(
                    meta,
                    "skipped",
                    Some("conflicting shader group descriptors".into()),
                )?;
                continue;
            }
            // Code chunks may be unnamed (retoc/src/shader_library.rs:1172-1173).
            // --pak-all also restores raw discovery for code chunks
            // without a same-container descriptor. Parsed groups stay bounded.
            if !(self.pak_all || toc.shader_code(chunk) || group.is_some() || named) {
                self.iostore_event(meta, "not_selected", None)?;
                continue;
            }
            if let Some((group, library)) = group {
                meta["shader_group"] = json!({"shaders": group.shaders.len(), "compressed_size": group.compressed,
                    "uncompressed_size": group.uncompressed});
                let mut data = Vec::new();
                if let Err(error) = toc.extract(path, chunk, self.pak_key.as_ref(), &mut data) {
                    let reason = format!("shader bytes unusable: group extraction failed: {error}");
                    self.iostore_event(meta.clone(), "skipped", Some(reason.clone()))?;
                    self.ue_group_skipped(source, &meta, library, group, &reason)?;
                    continue;
                }
                let (data, method) = match shaders::expand(&data, group) {
                    Ok(v) => v,
                    Err(error) => {
                        let reason = format!("shader group decompression failed: {error}");
                        self.iostore_event(meta.clone(), "raw-scanned", Some(reason.clone()))?;
                        // retoc/src/shader_library.rs:287-305: shader offsets
                        // address expanded bytes. On failure those views are
                        // unusable; scan the stored group once instead.
                        self.ue_group_skipped(source, &meta, library, group,
                            &format!("shader bytes unusable: {reason}; expanded shader boundaries unavailable; raw-scanning stored group"))?;
                        self.ue_blob(
                            source,
                            source_bytes.clone(),
                            json!({"iostore": meta, "library_chunk_id": library["chunk_id"]}),
                            Err(reason),
                            &data,
                        )?;
                        continue;
                    }
                };
                meta["shader_group"]["compression"] = json!(method);
                self.iostore_event(meta.clone(), "scanned", None)?;
                for shader in &group.shaders {
                    let context = json!({"iostore": meta, "library_chunk_id": library["chunk_id"], "shader": shader.metadata()});
                    let start = shader.offset as usize;
                    self.ue_blob(
                        source,
                        source_bytes.clone(),
                        context,
                        shader.stage().map_err(|e| e.to_string()),
                        &data[start..start + shader.size as usize],
                    )?;
                }
                continue;
            }
            if toc.shader_code(chunk) && !self.pak_all {
                self.iostore_event(meta, "skipped", Some("shader code has no supported same-container library descriptor; per-shader boundaries and frequency are unknown".into()))?;
                continue;
            }
            let (pending, file) = PendingFile::new(&self.output.join("iostore-chunk"))?;
            let mut writer = BufWriter::new(file);
            if let Err(error) = toc.extract(path, chunk, self.pak_key.as_ref(), &mut writer) {
                self.iostore_event(meta, "skipped", Some(error.to_string()))?;
                continue;
            }
            writer.flush()?;
            drop(writer);
            if toc.shader_code(chunk) {
                self.iostore_event(meta.clone(), "raw-scanned", Some(
                    "shader code has no supported same-container library descriptor; per-shader boundaries and frequency are unknown".into(),
                ))?;
                self.records(
                    &pending.temporary,
                    source,
                    source_bytes.clone(),
                    RecordSource::IoStore(&meta),
                )?;
            } else if chunk
                .name
                .as_ref()
                .is_some_and(|name| crate::ue_shaderlib::named(name))
            {
                self.ue_library(
                    source,
                    source_bytes.clone(),
                    json!({"iostore": meta}),
                    &pending.temporary,
                )?;
            } else {
                self.iostore_event(meta.clone(), "scanned", None)?;
                self.records(
                    &pending.temporary,
                    source,
                    source_bytes.clone(),
                    RecordSource::IoStore(&meta),
                )?;
            }
        }
        let chunk_ids: HashSet<_> = toc.chunks.iter().map(|chunk| chunk.id).collect();
        let mut missing: Vec<_> = groups
            .keys()
            .filter(|id| !chunk_ids.contains(*id))
            .copied()
            .collect();
        missing.sort();
        for id in missing {
            let (group, library) = &groups[&id];
            self.ue_group_skipped(source, &json!({"chunk_id": iostore::hex(&id)}), library, group,
                "shader group is absent from this container; cross-container library resolution is unsupported")?;
            self.iostore_event(json!({"kind": "iostore_chunk", "source_file": source,
                "source_path_hex": source_bytes, "chunk_id": iostore::hex(&id)}), "skipped",
                Some("shader group is absent from this container; cross-container library resolution is unsupported".into()))?;
        }
        Ok(())
    }
    fn ue_group_skipped(
        &mut self,
        source: &str,
        meta: &Value,
        library: &Value,
        group: &shaders::Group,
        reason: &str,
    ) -> io::Result<()> {
        for shader in &group.shaders {
            self.ue_event(source, json!({"iostore": meta, "library_chunk_id": library["chunk_id"], "shader": shader.metadata()}),
                "ue_shader", "skipped", Some(reason.into()))?;
        }
        Ok(())
    }

    fn iostore_event(
        &mut self,
        mut meta: Value,
        status: &str,
        reason: Option<String>,
    ) -> io::Result<()> {
        meta["status"] = json!(status);
        meta["reason"] = json!(reason);
        writeln!(
            self.report,
            "{}::chunk[{}]: {status}{}",
            meta["source_file"].as_str().unwrap(),
            meta["chunk_id"].as_str().unwrap(),
            reason.map(|r| format!(": {r}")).unwrap_or_default()
        )?;
        self.event(meta)
    }
}
fn chunk_meta(source: &str, source_bytes: &Value, toc: &Toc, index: usize) -> Value {
    let chunk = &toc.chunks[index];
    json!({"kind": "iostore_chunk", "source_file": source, "source_path_hex": source_bytes,
        "chunk_index": index, "chunk_id": iostore::hex(&chunk.id), "chunk_type": chunk.id[11],
        "entry_name": chunk.name, "logical_offset": chunk.offset, "length": chunk.length,
        "compression": toc.compression(chunk), "metadata_hash": chunk.hash, "metadata_flags": chunk.flags})
}
