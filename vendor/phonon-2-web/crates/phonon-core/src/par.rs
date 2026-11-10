//! Data parallelism that degrades to a plain loop: rayon when the `parallel` feature is on (native threads, or Web
//! Workers through `wasm-bindgen-rayon`), sequential otherwise.

/// Run `f(i)` for every `i in 0..n`, possibly on several threads.
pub fn for_each<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if n > 1 && rayon::current_num_threads() > 1 {
            (0..n).into_par_iter().for_each(f);
            return;
        }
    }
    (0..n).for_each(f);
}

/// `f(i)` for every `i in 0..n`, results in order.
pub fn map<T: Send, F: Fn(usize) -> T + Sync + Send>(n: usize, f: F) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if n > 1 && rayon::current_num_threads() > 1 {
            return (0..n).into_par_iter().map(f).collect();
        }
    }
    (0..n).map(f).collect()
}

/// Threads the parallel loops can use.
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// A raw pointer that may be shared between the workers of one parallel loop, each writing a disjoint region.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}
