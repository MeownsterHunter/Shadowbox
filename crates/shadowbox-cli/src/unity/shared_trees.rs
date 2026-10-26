//! Shared-tree format facts use the pinned UnityPy reference in
//! docs/reference-sources.md Independently written, no source port.
use super::*;

pub(super) fn read_shared_metadata(
    r: &mut Reader<'_>,
    version: u32,
    has_tree: bool,
    types: &mut [SerializedType],
    budget: &mut TreeBudget,
) -> io::Result<()> {
    // UnityPy/files/SerializedFile.py:318-338: scripts, externals, reference
    // types, then the v26 shared subtree table (also present without trees).
    // :47-53: each script is i32 file index and aligned i64 identifier.
    for _ in 0..r.count()? {
        r.u32()?;
        r.align(4)?;
        r.u64()?;
    }
    // :79-85: external empty cstring, 16-byte GUID, i32 type, path cstring.
    for _ in 0..r.count()? {
        r.cstr()?;
        r.take(16)?;
        r.u32()?;
        r.cstr()?;
    }
    let mut reference_types = Vec::new();
    for _ in 0..r.count()? {
        reference_types.push(read_type_kind(r, version, budget, has_tree, true)?);
    }
    let mut shared = BTreeMap::new();
    for _ in 0..r.count()? {
        let (hash, tree) = read_tree_handler(r, budget)?;
        if shared.insert(hash, tree).is_some() {
            return Err(invalid("duplicate Unity shared subtree hash"));
        }
    }
    // :343-344: user information follows the table.
    r.cstr()?;
    if r.at != r.data.len() {
        return Err(invalid("Unity metadata size mismatch"));
    }
    let mut cache = BTreeMap::new();
    for ty in types.iter_mut().chain(reference_types.iter_mut()) {
        ty.nodes = resolve(
            &ty.nodes,
            &ty.references,
            &shared,
            budget,
            &mut BTreeSet::new(),
            &mut cache,
        )?;
    }
    Ok(())
}

type Shared = BTreeMap<[u8; 16], TreeInfo>;
type Cache = BTreeMap<[u8; 16], Vec<Node>>;

fn append(out: &mut Vec<Node>, node: &Node, level: u8, budget: &mut TreeBudget) -> io::Result<()> {
    // Expanded copies share the file's original node/string budgets, bounding
    // repeated references as well as the encoded tables. This is local policy.
    budget.nodes = budget
        .nodes
        .checked_sub(1)
        .ok_or_else(|| invalid("Unity expanded tree nodes exceed resource limit"))?;
    budget.strings = budget
        .strings
        .checked_sub(node.ty.len() + node.name.len())
        .ok_or_else(|| invalid("Unity expanded tree strings exceed resource limit"))?;
    let mut node = node.clone();
    node.level = level;
    node.shared = None;
    out.push(node);
    Ok(())
}

fn resolve(
    nodes: &[Node],
    hashes: &[[u8; 16]],
    shared: &Shared,
    budget: &mut TreeBudget,
    active: &mut BTreeSet<[u8; 16]>,
    cache: &mut Cache,
) -> io::Result<Vec<Node>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < nodes.len() {
        let node = &nodes[at];
        append(&mut out, node, node.level, budget)?;
        if let Some(index) = node.shared {
            // UnityPy/files/SerializedFile.py:369-384: m_RefTypeHash indexes
            // the wrapper hashes. Replace this node's children with the shared
            // root's children, retaining the referring node's name/type/flags.
            let index = usize::try_from(index)
                .map_err(|_| invalid("Unity shared reference index exceeds address space"))?;
            let hash = hashes
                .get(index)
                .ok_or_else(|| invalid("Unity shared reference index out of range"))?;
            if !cache.contains_key(hash) {
                let (children, references) = shared
                    .get(hash)
                    .ok_or_else(|| invalid("missing Unity shared subtree hash"))?;
                if !active.insert(*hash) {
                    return Err(invalid("cyclic Unity shared subtree reference"));
                }
                // u8 levels cannot describe a deeper expanded hierarchy. Bound
                // reference recursion even when a chain has no local children.
                if active.len() > 256 {
                    return Err(invalid("Unity shared subtree depth exceeds resource limit"));
                }
                let expanded = resolve(children, references, shared, budget, active, cache)?;
                active.remove(hash);
                cache.insert(*hash, expanded);
            }
            for child in cache[hash].iter().skip(1) {
                let level = node
                    .level
                    .checked_add(child.level)
                    .ok_or_else(|| invalid("Unity expanded tree depth overflow"))?;
                append(&mut out, child, level, budget)?;
            }
            at = node.end;
        } else {
            at += 1;
        }
    }
    // Rebuild flat subtree boundaries after insertion. Hierarchy validation
    // uses the same depth rule as AssetStudio/TypeTreeHelper.cs:308-323.
    let mut stack: Vec<usize> = Vec::new();
    let count = out.len();
    for at in 0..count {
        let level = out[at].level;
        while stack.last().is_some_and(|&i| out[i].level >= level) {
            out[stack.pop().unwrap()].end = at;
        }
        if (at == 0 && level != 0)
            || (at != 0 && stack.last().is_none_or(|&i| out[i].level + 1 != level))
        {
            return Err(invalid("invalid Unity expanded tree hierarchy"));
        }
        out[at].end = count;
        stack.push(at);
    }
    Ok(out)
}
