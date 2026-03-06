use crate::action_tree::*;
use crate::card::NOT_DEALT;
use crate::interface::GameNode;
use crate::mutex_like::{MutexGuardLike, MutexLike};
use std::ptr;
use std::slice;

use super::PostFlopNode;

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            prev_action: Action::None,
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            is_locked: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            num_elements_ip: 0,
            num_elements: 0,
            scale1: 0.0,
            scale2: 0.0,
            scale3: 0.0,
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            storage3: ptr::null_mut(),
        }
    }
}

impl PostFlopNode {
    /// Returns the child nodes as a slice.
    ///
    /// # Safety
    ///
    /// This relies on the node arena being a contiguous `Vec<MutexLike<PostFlopNode>>` where
    /// `children_offset` is a valid offset from `self` to the first child. `MutexLike<T>` is
    /// `#[repr(transparent)]`, so the pointer cast is layout-compatible.
    #[inline]
    pub(crate) fn children(&self) -> &[MutexLike<Self>] {
        // SAFETY: `MutexLike<T>` is `#[repr(transparent)]` around `UnsafeCell<T>`,
        // and nodes are stored contiguously in `PostFlopGame::node_arena`.
        // `children_offset` is set during tree construction to point at the correct
        // position relative to `self`.
        let self_ptr = self as *const Self as *const MutexLike<PostFlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }
}

impl GameNode for PostFlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }

    #[inline]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        let prev_player = self.player & PLAYER_MASK;
        match prev_player {
            0 => Some(1),
            1 => Some(0),
            _ => None,
        }
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<'_, Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        // SAFETY: `storage1` points into `PostFlopGame::storage1`, which is a `Vec<u8>`
        // allocated with proper alignment for f32. `num_elements` is set during tree
        // construction to match the allocated slice length.
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `strategy()`.
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        // SAFETY: `storage2` points into `PostFlopGame::storage2`. Same alignment/length
        // invariants as `strategy()`.
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `regrets()`.
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        // SAFETY: counterfactual values share `storage2` with regrets (mutually exclusive
        // phases of the solve). Same invariants apply.
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `cfvalues()`.
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn has_cfvalues_ip(&self) -> bool {
        self.num_elements_ip != 0
    }

    #[inline]
    fn cfvalues_ip(&self) -> &[f32] {
        // SAFETY: `storage3` points into `PostFlopGame::storage_ip`. `num_elements_ip` is
        // the correct length for this slice, set during tree construction.
        unsafe {
            slice::from_raw_parts(self.storage3 as *const f32, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `cfvalues_ip()`.
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut f32, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance(&self) -> &[f32] {
        // SAFETY: For chance nodes, counterfactual values are stored in `storage1`
        // (the strategy slot is unused at chance nodes).
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `cfvalues_chance()`.
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    // -- Compressed variants (cast to u16/i16 instead of f32) --

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        // SAFETY: When compression is enabled, `storage1` holds `u16` values instead of
        // `f32`. The allocation size matches `num_elements * size_of::<u16>()`.
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        // SAFETY: same as `strategy_compressed()`.
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        // SAFETY: Compressed regrets use `i16` in `storage2`.
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        // SAFETY: same as `regrets_compressed()`.
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        // SAFETY: Compressed cfvalues share `storage2` (same as regrets).
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        // SAFETY: same as `cfvalues_compressed()`.
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        // SAFETY: Compressed IP cfvalues in `storage3`.
        unsafe {
            slice::from_raw_parts(self.storage3 as *const i16, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        // SAFETY: same as `cfvalues_ip_compressed()`.
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut i16, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        // SAFETY: Compressed chance cfvalues in `storage1`.
        unsafe { slice::from_raw_parts(self.storage1 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        // SAFETY: same as `cfvalues_chance_compressed()`.
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut i16, self.num_elements as usize) }
    }

    // -- Scale factors --

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_ip_scale(&self) -> f32 {
        self.scale3
    }

    #[inline]
    fn set_cfvalue_ip_scale(&mut self, scale: f32) {
        self.scale3 = scale;
    }

    #[inline]
    fn cfvalue_chance_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_cfvalue_chance_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_default_values() {
        let node = PostFlopNode::default();
        assert_eq!(node.prev_action, Action::None);
        assert_eq!(node.player, PLAYER_OOP);
        assert_eq!(node.turn, NOT_DEALT);
        assert_eq!(node.river, NOT_DEALT);
        assert!(!node.is_locked);
        assert_eq!(node.amount, 0);
        assert_eq!(node.children_offset, 0);
        assert_eq!(node.num_children, 0);
        assert_eq!(node.num_elements_ip, 0);
        assert_eq!(node.num_elements, 0);
        assert_eq!(node.scale1, 0.0);
        assert_eq!(node.scale2, 0.0);
        assert_eq!(node.scale3, 0.0);
        assert!(node.storage1.is_null());
        assert!(node.storage2.is_null());
        assert!(node.storage3.is_null());
    }

    #[test]
    fn node_terminal_flags() {
        let mut node = PostFlopNode::default();
        node.player = PLAYER_OOP;
        assert!(!node.is_terminal());
        assert!(!node.is_chance());
        assert_eq!(node.player(), 0);

        node.player = PLAYER_IP;
        assert!(!node.is_terminal());
        assert!(!node.is_chance());
        assert_eq!(node.player(), 1);

        node.player = PLAYER_TERMINAL_FLAG;
        assert!(node.is_terminal());
        assert!(!node.is_chance());

        node.player = PLAYER_FOLD_FLAG;
        assert!(node.is_terminal());
        assert!(!node.is_chance());

        node.player = PLAYER_CHANCE_FLAG | PLAYER_OOP;
        assert!(!node.is_terminal());
        assert!(node.is_chance());
    }

    #[test]
    fn node_cfvalue_storage_player() {
        let mut node = PostFlopNode::default();

        node.player = PLAYER_OOP;
        assert_eq!(node.cfvalue_storage_player(), Some(1));

        node.player = PLAYER_IP;
        assert_eq!(node.cfvalue_storage_player(), Some(0));

        node.player = PLAYER_CHANCE_FLAG | PLAYER_CHANCE;
        assert_eq!(node.cfvalue_storage_player(), None);
    }

    #[test]
    fn node_has_cfvalues_ip() {
        let mut node = PostFlopNode::default();
        assert!(!node.has_cfvalues_ip());
        node.num_elements_ip = 10;
        assert!(node.has_cfvalues_ip());
    }

    #[test]
    fn node_enable_parallelization() {
        let mut node = PostFlopNode::default();
        // Default river is NOT_DEALT, so parallelization is enabled.
        assert!(node.enable_parallelization());

        node.river = 5;
        assert!(!node.enable_parallelization());
    }

    #[test]
    fn node_scale_factors() {
        let mut node = PostFlopNode::default();
        assert_eq!(node.strategy_scale(), 0.0);
        assert_eq!(node.regret_scale(), 0.0);
        assert_eq!(node.cfvalue_ip_scale(), 0.0);

        node.set_strategy_scale(1.5);
        assert_eq!(node.strategy_scale(), 1.5);
        // strategy and cfvalue_chance share scale1
        assert_eq!(node.cfvalue_chance_scale(), 1.5);

        node.set_regret_scale(2.0);
        assert_eq!(node.regret_scale(), 2.0);
        // regret and cfvalue share scale2
        assert_eq!(node.cfvalue_scale(), 2.0);

        node.set_cfvalue_ip_scale(3.0);
        assert_eq!(node.cfvalue_ip_scale(), 3.0);
    }

    #[test]
    fn node_num_actions() {
        let mut node = PostFlopNode::default();
        assert_eq!(node.num_actions(), 0);
        node.num_children = 3;
        assert_eq!(node.num_actions(), 3);
    }

    #[test]
    fn node_repr_c_size() {
        // Verify that the #[repr(C)] layout produces a predictable size.
        // On 64-bit: Action(8) + u8(1) + Card(1) + Card(1) + bool(1) + pad(4) + i32(4) +
        //            u32(4) + u16(2) + u16(2) + u32(4) + f32(4) + f32(4) + f32(4) +
        //            ptr(8) + ptr(8) + ptr(8) = ~72 bytes (with repr(C) padding)
        let size = std::mem::size_of::<PostFlopNode>();
        // Just verify it's a reasonable size (not zero, not absurdly large).
        assert!(size > 0 && size <= 128, "unexpected size: {size}");
    }
}
