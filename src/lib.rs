#![no_std]

#![deny(missing_debug_implementations)]

#![feature(asm)]
#![feature(const_fn)]
#![feature(const_let)]
#![feature(nll)]

#[macro_use]
extern crate bitflags;
extern crate real_memory;

use core::fmt;

pub use real_memory::Address as PhysAddr;

pub const LOG_PAGE_SIZE : u32 = 12;
pub const LOG_TABLE_SIZE: u32 = LOG_PAGE_SIZE - LOG_ENTRY_SIZE;

#[cfg(any(target_arch = "x86_64", target_arch = "riscv"))]
const LOG_ENTRY_SIZE: u32 = 3;

#[cfg(any(target_arch = "x86_64", target_arch = "riscv"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Entry(u64);

impl Entry {
    pub const Empty: Self = Entry(0);

    #[inline]
    pub const fn new(PhysFrame(PhysAddr(a)): PhysFrame, flags: EntryFlags) -> Self {
        Entry(a | flags.bits as u64)
    }

    #[inline]
    pub fn flags(self) -> EntryFlags { EntryFlags::from_bits_truncate(self.0 as _) }

    #[inline]
    pub fn frame(self) -> Result<PhysFrame, FrameError> {
        if !self.flags().contains(EntryFlags::Present) { Err(FrameError::Absent) }
        else { Ok(From::from(PhysAddr(self.0))) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameError { Absent, Huge }

#[cfg(target_arch = "x86_64")]
bitflags! {
    pub struct EntryFlags: u16 {
        const Present   = 1 <<  0;
        const Write     = 1 <<  1;
        const User      = 1 <<  2;
        const Writethru = 1 <<  3;
        const NoCache   = 1 <<  4;
        const Accessed  = 1 <<  5;
        const Dirty     = 1 <<  6;
        const HugePage  = 1 <<  7;
        const Global    = 1 <<  8;
        const Custom1   = 1 <<  9;
        const Custom2   = 1 << 10;
        const Custom3   = 1 << 11;
        const Custom    = Self::Custom1.bits
                        | Self::Custom2.bits
                        | Self::Custom3.bits;
    }
}

#[cfg(target_arch = "riscv")]
bitflags! {
    pub struct EntryFlags: u16 {
        const Present   = 1 <<  0;
        const Read      = 1 <<  1;
        const Write     = 1 <<  2;
        const Execute   = 1 <<  3;
        const User      = 1 <<  4;
        const Global    = 1 <<  5;
        const Accessed  = 1 <<  6;
        const Dirty     = 1 <<  7;
        const Custom1   = 1 <<  8;
        const Custom2   = 1 <<  9;
        const Custom    = Custom1.bits
                        | Custom2.bits;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysFrame(PhysAddr);

impl From<PhysAddr> for PhysFrame {
    #[inline]
    fn from(PhysAddr(a): PhysAddr) -> Self { PhysFrame(PhysAddr((a >> LOG_PAGE_SIZE) << LOG_PAGE_SIZE)) }
}

#[derive(Clone, Copy)]
#[repr(align(0x1000))]
pub struct Table(pub [Entry; 1 << LOG_TABLE_SIZE]);

impl fmt::Debug for Table {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.0[..].fmt(fmt)
    }
}

impl Table {
    pub const Empty: Self = Table([Entry::Empty; 1 << LOG_TABLE_SIZE]);
}

#[derive(Debug)]
pub struct RecursiveTable<'a> {
    table: &'a mut Table,
    recursive_index: usize,
}

impl<'a> RecursiveTable<'a> {
    #[inline]
    pub fn new(table: &'a mut Table) -> Result<Self, ()> {
        let page = Page::from(VirtAddr(table as *mut _ as _));
        let recursive_index = page.p4_index();

        if page.p3_index() != recursive_index ||
           page.p2_index() != recursive_index ||
           page.p1_index() != recursive_index ||
           Ok(active_table().0) != table.0[recursive_index].frame() { return Err(()) }

        Ok(Self { table, recursive_index })
    }

    #[inline]
    pub unsafe fn new_unchecked(table: &'a mut Table, recursive_index: usize) -> Self {
        Self { table, recursive_index }
    }

    unsafe fn create_next_table<'b, A: FrameAlloc>(entry: &'b mut Entry, next_table_page: Page, a: &mut A) -> Result<&'b mut Table, MapError> {
        let was_created = if 0 == entry.0 {
            let frame = a.alloc().ok_or(MapError::FrameAllocationFailed)?;
            *entry = Entry::new(frame, EntryFlags::Present | EntryFlags::Write);
            true
        } else { false };
        if entry.flags().contains(EntryFlags::HugePage) {
            return Err(MapError::ParentEntryHugePage);
        }
        let table: &mut Table = &mut *((next_table_page.0).0 as *mut _);
        if was_created { table.0 = [Entry(0); 1 << LOG_TABLE_SIZE] }
        Ok(table)
    }

    pub fn translate_page(&self, page: Page) -> Option<PhysFrame> {
        let mut entry = Entry(0);
        for k in (1..5).rev() {
            let table = if 4 == k { &self.table }
                        else { unsafe { &*pn_ptr(k, page, self.recursive_index) } };
            entry = table.0[page.0.page_table_index(k as _)];
            if 0 == entry.0 { return None }
        }
        entry.frame().ok()
    }

    pub fn map<A: FrameAlloc>(&mut self, page: Page, frame: PhysFrame, flags: EntryFlags, a: &mut A) -> Result<MapperFlush, MapError> {
        let mut table = &mut *self.table;
        for k in (1..4).rev() {
            let page = pn_page(k, page, self.recursive_index);
            table = unsafe { Self::create_next_table(&mut table.0[page.0.page_table_index(k as u32+1)], page, a) }?;
        }
        if 0 != table.0[page.p1_index()].0 { return Err(MapError::PageAlreadyMapped) }
        table.0[page.p1_index()] = Entry::new(frame, flags);
        Ok(MapperFlush(page))
    }

    pub fn unmap(&mut self, page: Page) -> Result<(PhysFrame, MapperFlush), UnmapError> {
        let mut frame = PhysFrame(PhysAddr(0));
        for k in (1..5).rev() {
            let table = if 4 == k { &mut *self.table }
                        else { unsafe { &mut *pn_ptr(k, page, self.recursive_index) } };
            let entry = &mut table.0[page.0.page_table_index(k as _)];
            frame = entry.frame().map_err(|e| match e {
                FrameError::Absent => UnmapError::PageNotMapped,
                FrameError::Huge => UnmapError::ParentEntryHugePage,
            })?;
            if 1 == k { *entry = Entry(0) }
        }
        Ok((frame, MapperFlush(page)))
    }

    pub fn identity_map<A: FrameAlloc>(&mut self, frame: PhysFrame, flags: EntryFlags, a: &mut A) -> Result<MapperFlush, IdentityMapError> {
        let page = Page(VirtAddr((frame.0).0 as _));
        if (page.0).0 as u64 != (frame.0).0 { return Err(IdentityMapError::BadAddress) }
        self.map(page, frame, flags, a).map_err(IdentityMapError::MapError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapperFlush(Page);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapError {
    FrameAllocationFailed,
    ParentEntryHugePage,
    PageAlreadyMapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityMapError {
    BadAddress,
    MapError(MapError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnmapError {
    ParentEntryHugePage,
    PageNotMapped,
}

#[inline]
fn pn_ptr(lvl: usize, page: Page, rec_ix: usize) -> *mut Table {
    (pn_page(lvl, page, rec_ix).0).0 as _
}

fn pn_page(lvl: usize, page: Page, rec_ix: usize) -> Page {
    let ixs = [rec_ix, rec_ix, rec_ix, page.p4_index(), page.p3_index(), page.p2_index()];
    let ixs = &ixs[3-lvl..];
    Page::from_page_table_indices([ixs[0], ixs[1], ixs[2], ixs[3]])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VirtAddr(usize);

unsafe impl Send for VirtAddr {}
unsafe impl Sync for VirtAddr {}

impl VirtAddr {
    #[inline]
    pub const fn page_offset(self) -> usize { self.0 & ((1 << LOG_PAGE_SIZE) - 1) }

    #[inline]
    pub const fn page_table_index(self, n: u32) -> usize { self.0 >> (LOG_TABLE_SIZE*n) & ((1 << LOG_TABLE_SIZE) - 1) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Page(VirtAddr);

impl From<VirtAddr> for Page {
    #[inline]
    fn from(VirtAddr(a): VirtAddr) -> Self { Page(VirtAddr((a >> LOG_PAGE_SIZE) << LOG_PAGE_SIZE)) }
}

impl Page {
    #[inline]
    const fn p4_index(self) -> usize { self.0.page_table_index(4) }

    #[inline]
    const fn p3_index(self) -> usize { self.0.page_table_index(3) }

    #[inline]
    const fn p2_index(self) -> usize { self.0.page_table_index(2) }

    #[inline]
    const fn p1_index(self) -> usize { self.0.page_table_index(1) }

    #[inline]
    const fn from_page_table_indices(ixs: [usize; 4]) -> Self {
        Page(VirtAddr(ixs[0] << LOG_PAGE_SIZE + 3*LOG_TABLE_SIZE |
                      ixs[1] << LOG_PAGE_SIZE + 2*LOG_TABLE_SIZE |
                      ixs[2] << LOG_PAGE_SIZE + 1*LOG_TABLE_SIZE |
                      ixs[3] << LOG_PAGE_SIZE + 0*LOG_TABLE_SIZE))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn active_table() -> (PhysFrame, TableFlags) {
    let value: u64;
    unsafe { asm!("mov $0, cr3" : "=r" (value)); }
    (PhysFrame::from(PhysAddr(value)), TableFlags(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableFlags(());

pub trait FrameAlloc {
    fn alloc(&mut self) -> Option<PhysFrame>;
}
