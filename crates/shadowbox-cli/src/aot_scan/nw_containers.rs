use super::*;
use crate::nw;

// Cumulative limits bound overlapping ranges, archive fanout and decompression
// bombs even when every individual file is under MAX_BYTES.
#[derive(Default)]
pub(super) struct NwBudget {
    bytes: usize,
    nodes: usize,
}
impl NwBudget {
    fn visit(&mut self, bytes: usize, depth: usize) -> nw::Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or("container work overflow")?;
        self.nodes += 1;
        if depth >= nw::MAX_DEPTH || self.bytes > 1024 * 1024 * 1024 || self.nodes > 65536 {
            return Err("container recursion/work limit".into());
        }
        Ok(())
    }
}

fn dictionary_discovery(
    name: &str,
    bytes: &[u8],
    dictionaries: &mut nw::Dictionaries,
    depth: usize,
    budget: &mut NwBudget,
) -> nw::Result<()> {
    budget.visit(bytes.len(), depth)?;
    // A suffix is a hint for a raw dictionary only after wrappers are removed.
    match nw::kind(bytes) {
        Some("Yaz0") => {
            dictionary_discovery(name, &nw::yaz0(bytes)?, dictionaries, depth + 1, budget)
        }
        Some("zstd") => {
            let (data, _) = dictionaries.zstd(bytes)?;
            dictionary_discovery(
                name.strip_suffix(".zs").unwrap_or(name),
                &data,
                dictionaries,
                depth + 1,
                budget,
            )
        }
        Some("SARC") => {
            let sarc = nw::sarc(bytes)?;
            for entry in sarc.entries {
                // Continue past a sibling needing a dictionary, so other entries
                // can provide it on this pass. Retry on the next fixed-point pass.
                let _ = dictionary_discovery(
                    &format!("{name}::{}", entry.name),
                    &bytes[entry.range],
                    dictionaries,
                    depth + 1,
                    budget,
                );
            }
            Ok(())
        }
        _ if nw::dictionary_hint(name, bytes) => dictionaries.add(name, bytes),
        _ => Ok(()),
    }
}

impl Scanner<'_> {
    pub(super) fn load_dictionaries(&mut self) -> io::Result<()> {
        // Iterate to a fixed point: dictionary packs may themselves depend on
        // another dictionary. Bytes are deduplicated; conflicting IDs are kept
        // so the decoder reports ambiguity. No archive entry is a host pathname.
        loop {
            let before = self.dictionaries.items.len();
            let mut directories = vec![super::sorted_directory(&self.root)?];
            let mut budget = NwBudget::default();
            while let Some(directory) = directories.last_mut() {
                let Some(entry) = directory.next() else {
                    directories.pop();
                    continue;
                };
                let path = entry;
                if path == self.output || self.cache_dir.as_ref() == Some(&path) {
                    continue;
                }
                let kind = fs::symlink_metadata(&path)?.file_type();
                if kind.is_dir() {
                    directories.push(super::sorted_directory(&path)?);
                } else if kind.is_file() {
                    let mut file = File::open(&path)?;
                    let mut prefix = [0; 16];
                    let n = file.read(&mut prefix)?;
                    let name = path.strip_prefix(&self.root).unwrap().to_string_lossy();
                    if nw::dictionary_hint(&name, &prefix[..n])
                        || matches!(nw::kind(&prefix[..n]), Some("SARC" | "Yaz0" | "zstd"))
                    {
                        if budget.bytes >= 1024 * 1024 * 1024 || budget.nodes >= 65536 {
                            break;
                        }
                        if file.metadata()?.len() > nw::MAX_BYTES as u64 {
                            continue;
                        }
                        file.seek(SeekFrom::Start(0))?;
                        let mut bytes = Vec::new();
                        file.take(nw::MAX_BYTES as u64 + 1)
                            .read_to_end(&mut bytes)?;
                        if bytes.len() <= nw::MAX_BYTES {
                            let _ = dictionary_discovery(
                                &name,
                                &bytes,
                                &mut self.dictionaries,
                                0,
                                &mut budget,
                            );
                        }
                    }
                }
            }
            if self.dictionaries.items.len() == before {
                break;
            }
        }
        let dictionaries: Vec<_> = self.dictionaries.items.iter().map(|d| json!({"source": d.name, "id": d.id, "size": d.bytes.len(), "sha256": format!("{:x}", Sha256::digest(&d.bytes))})).collect();
        self.event(json!({"kind": "nw_dictionaries", "dictionaries": dictionaries}))
    }

    pub(super) fn nw_skipped(
        &mut self,
        source: &str,
        path: &Value,
        reason: impl ToString,
    ) -> io::Result<()> {
        self.totals.nw_skipped += 1;
        let reason = reason.to_string();
        writeln!(self.report, "{source}::{path}: container skipped: {reason}")?;
        self.event(json!({"kind": "nw_skipped", "source_file": source, "container_path": path, "reason": reason}))
    }

    pub(super) fn nw(
        &mut self,
        source: &str,
        bytes: &[u8],
        path: &[Value],
        budget: &mut NwBudget,
    ) -> io::Result<()> {
        if let Err(error) = budget.visit(bytes.len(), path.len()) {
            return self.nw_skipped(source, &json!(path), error);
        }
        let Some(kind) = nw::kind(bytes) else {
            let name = path
                .iter()
                .rev()
                .find_map(|v| v["entry"].as_str())
                .unwrap_or(source);
            let name = name.strip_suffix(".zs").unwrap_or(name);
            if nw::dictionary_hint(name, bytes) {
                let loaded = self.dictionaries.items.iter().any(|d| d.bytes == bytes);
                return self.event(json!({"kind": "nw_dictionary_file", "source_file": source, "container_path": path, "size": bytes.len(), "loaded": loaded, "reason": if loaded { None } else { Some("dictionary was invalid or exceeded discovery limits") }}));
            }
            // Ordinary members retain the existing raw-record discovery policy.
            let context = NwStage {
                meta: json!({"container_path": path}),
                stage: None,
            };
            let (pending, mut file) = PendingFile::new(&self.output.join("nw-member"))?;
            file.write_all(bytes)?;
            drop(file);
            return self.records(
                &pending.temporary,
                source,
                Value::Null,
                RecordSource::Nw(&context),
            );
        };
        self.totals.nw_containers += 1;
        let before = (
            self.totals.accepted,
            self.totals.rejected,
            self.totals.nw_stage_failed,
            self.totals.nw_skipped,
            self.totals.nw_variation_skipped,
        );
        let mut child = path.to_vec();
        child.push(json!({"format": kind}));
        match kind {
            "Yaz0" | "zstd" => {
                let result = if kind == "Yaz0" {
                    nw::yaz0(bytes).map(|data| (data, Vec::new()))
                } else {
                    self.dictionaries.zstd(bytes)
                };
                let (data, dictionaries) = match result {
                    Ok(v) => v,
                    Err(e) => return self.nw_skipped(source, &json!(child), e),
                };
                self.event(json!({"kind": "nw_compression", "source_file": source, "container_path": child, "format": kind, "compressed_size": bytes.len(), "decoded_size": data.len(), "dictionaries": dictionaries}))?;
                self.nw(source, &data, &child, budget)?;
            }
            "SARC" => {
                let sarc = match nw::sarc(bytes) {
                    Ok(v) => v,
                    Err(e) => return self.nw_skipped(source, &json!(child), e),
                };
                self.event(json!({"kind": "nw_sarc", "source_file": source, "container_path": child, "version": 0x100, "big_endian": sarc.big_endian, "hash_multiplier": sarc.multiplier, "entries": sarc.entries.len()}))?;
                for (index, entry) in sarc.entries.iter().enumerate() {
                    let mut entry_path = child.clone();
                    entry_path.push(json!({"entry": entry.name, "index": index, "name_hex": entry.name_hex, "hash": entry.hash, "attributes": entry.attributes, "offset": entry.range.start, "size": entry.range.len()}));
                    self.event(json!({"kind": "nw_sarc_entry", "source_file": source, "container_path": entry_path}))?;
                    self.nw(source, &bytes[entry.range.clone()], &entry_path, budget)?;
                }
            }
            "BFSHA" | "BFRES" | "BNTX" => {
                let header = match nw::wrapper(bytes, kind) {
                    Ok(v) => v,
                    Err(e) => return self.nw_skipped(source, &json!(child), e),
                };
                self.event(json!({"kind": "nw_wrapper", "source_file": source, "container_path": child, "format": kind, "version": header.version, "size": header.size}))?;
                // Discovery policy: search the declared wrapper extent for
                // independently validated binary children. BfshaLoader.cs:114-119
                // supplies BNSH's self-contained size/substream, not a shader
                // instruction mapping. BFRES ExternalFile.cs:7,34-38 permits
                // arbitrary attachments. BNTX permits opaque byte user data
                // (BntxLibrary/BntxReader.cs:97-104,139-145). This is bounded
                // child discovery, not texture/model table interpretation.
                let mut at = 32;
                while at + 4 <= header.size {
                    let tail = &bytes[at..header.size];
                    // Treat every recognized binary child as a boundary, even
                    // when its version is unknown. Never scan through that
                    // child's extent and bypass its version rejection.
                    if matches!(nw::kind(tail), Some("BNSH" | "BFSHA" | "BFRES" | "BNTX")) {
                        let mut embedded = child.clone();
                        embedded.push(json!({"embedded_offset": at}));
                        match nw::binary_header(tail) {
                            Ok(h) => {
                                self.nw(source, &tail[..h.size], &embedded, budget)?;
                                at += h.size;
                                continue;
                            }
                            Err(e) => self.nw_skipped(source, &json!(embedded), e)?,
                        }
                    }
                    at += 1;
                }
            }
            "BNSH" => {
                let bnsh = match nw::bnsh(bytes) {
                    Ok(v) => v,
                    Err(e) => return self.nw_skipped(source, &json!(child), e),
                };
                self.event(json!({"kind": "nw_bnsh", "source_file": source, "container_path": child, "version": bnsh.header.version, "size": bnsh.header.size, "variations": bnsh.variations.len()}))?;
                for (index, variation) in bnsh.variations.into_iter().enumerate() {
                    self.totals.nw_variations += 1;
                    let variation = match variation {
                        Ok(v) => v,
                        Err(error) => {
                            self.totals.nw_variation_skipped += 1;
                            writeln!(
                                self.report,
                                "{source}::{}::variation[{index}]: skipped: {error}",
                                json!(child)
                            )?;
                            self.event(json!({"kind": "nw_variation", "source_file": source, "container_path": child, "variation": index, "status": "skipped", "reason": error}))?;
                            continue;
                        }
                    };
                    let before = (
                        self.totals.accepted,
                        self.totals.rejected,
                        self.totals.nw_stage_failed,
                    );
                    for stage in &variation.stages {
                        // Preserve both exact ranges even if raw layout validation
                        // fails. Control bytes remain opaque; no fake shader header.
                        let code = &bytes[stage.code.clone()];
                        let control = &bytes[stage.control.clone()];
                        // Shared/overlapping stage pointers still cost work and
                        // output I/O. Charge each occurrence, not just the parent.
                        if let Err(error) = budget.visit(code.len() + control.len(), child.len()) {
                            self.totals.nw_stage_failed += 1;
                            self.totals.nw_variation_skipped += 1;
                            self.event(json!({"kind": "nw_stage", "source_file": source, "nw": {"container_path": child, "variation": index, "stage": stage_name(stage.stage)}, "status": "failed", "reason": error}))?;
                            self.event(json!({"kind": "nw_variation", "source_file": source, "container_path": child, "variation": index, "status": "skipped", "reason": error}))?;
                            return self.nw_skipped(source, &json!(child), error);
                        }
                        let code_name = format!("{:x}.nw-code.bin", Sha256::digest(code));
                        let control_name = format!("{:x}.nw-control.bin", Sha256::digest(control));
                        write_asset(&self.output.join(&code_name), code)?;
                        write_asset(&self.output.join(&control_name), control)?;
                        let context = NwStage {
                            stage: Some(stage.stage),
                            meta: json!({"container_path": child, "variation": index, "variation_offset": variation.offset, "binary_offset": variation.binary_offset, "stage": stage_name(stage.stage), "descriptor_offset": stage.descriptor, "code_offset": stage.code.start, "code_size": stage.code.len(), "control_offset": stage.control.start, "control_size": stage.control.len(), "code_output": code_name, "control_output": control_name}),
                        };
                        let counts = (
                            self.totals.candidates,
                            self.totals.accepted,
                            self.totals.rejected,
                        );
                        let (pending, mut file) = PendingFile::new(&self.output.join("nw-stage"))?;
                        file.write_all(code)?;
                        drop(file);
                        self.records(
                            &pending.temporary,
                            source,
                            Value::Null,
                            RecordSource::Nw(&context),
                        )?;
                        let candidates = self.totals.candidates - counts.0;
                        let accepted = self.totals.accepted - counts.1;
                        let rejected = self.totals.rejected - counts.2;
                        let status = if accepted == 0 || rejected != 0 {
                            self.totals.nw_stage_failed += 1;
                            "failed"
                        } else {
                            "scanned"
                        };
                        self.event(json!({"kind": "nw_stage", "source_file": source, "nw": context.meta, "status": status, "candidates": candidates, "accepted": accepted, "rejected": rejected, "reason": if candidates == 0 { Some("unsupported stage bytecode layout: no known program record") } else { None }}))?;
                    }
                    let accepted = self.totals.accepted - before.0;
                    let rejected = self.totals.rejected - before.1;
                    let failed = self.totals.nw_stage_failed - before.2;
                    writeln!(self.report, "{source}::{}::variation[{index}]: stages={} accepted={accepted} rejected={rejected} failed={failed}", json!(child), variation.stages.len())?;
                    self.event(json!({"kind": "nw_variation", "source_file": source, "container_path": child, "variation": index, "offset": variation.offset, "binary_offset": variation.binary_offset, "status": if failed > 0 { "failed" } else if variation.stages.is_empty() { "empty" } else { "scanned" }, "stages": variation.stages.len(), "accepted": accepted, "rejected": rejected, "failed_stages": failed}))?;
                }
            }
            _ => unreachable!(),
        }
        let accepted = self.totals.accepted - before.0;
        let rejected = self.totals.rejected - before.1;
        let failed = self.totals.nw_stage_failed - before.2;
        let skipped = self.totals.nw_skipped - before.3;
        let skipped_variations = self.totals.nw_variation_skipped - before.4;
        writeln!(self.report, "{source}::{}: {kind} accepted={accepted} rejected={rejected} failed_stages={failed} skipped={skipped} skipped_variations={skipped_variations}", json!(child))?;
        self.event(json!({"kind": "nw_container", "source_file": source, "container_path": child, "format": kind, "accepted": accepted, "rejected": rejected, "failed_stages": failed, "skipped": skipped, "skipped_variations": skipped_variations}))
    }
}
