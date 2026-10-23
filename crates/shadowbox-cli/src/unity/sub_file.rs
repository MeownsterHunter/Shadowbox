//! A seekable directory range in the decoded UnityFS spool. Bundle range facts:
//! AssetStudio/BundleFile.cs:298-308,317-357, pinned in 0017-unity.md.
use super::*;

pub(crate) struct SubFile<'a, R> {
    input: &'a mut R,
    origin: u64,
    length: u64,
    at: u64,
}
impl<'a, R: Read + Seek> SubFile<'a, R> {
    pub fn new(input: &'a mut R, range: Range<usize>) -> Self {
        Self {
            input,
            origin: range.start as u64,
            length: range.len() as u64,
            at: 0,
        }
    }
}
impl<R: Read + Seek> Read for SubFile<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = (self.length - self.at).min(bytes.len() as u64) as usize;
        self.input.seek(SeekFrom::Start(self.origin + self.at))?;
        let read = self.input.read(&mut bytes[..count])?;
        self.at += read as u64;
        Ok(read)
    }
}
impl<R: Read + Seek> Seek for SubFile<'_, R> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(at) => i128::from(at),
            SeekFrom::End(offset) => i128::from(self.length) + i128::from(offset),
            SeekFrom::Current(offset) => i128::from(self.at) + i128::from(offset),
        };
        if !(0..=i128::from(self.length)).contains(&position) {
            return Err(invalid("UnityFS subfile seek outside directory range"));
        }
        self.at = position as u64;
        Ok(self.at)
    }
}
