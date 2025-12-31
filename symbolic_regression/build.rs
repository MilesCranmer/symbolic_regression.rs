use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        wgpu: { all(feature = "wgpu", not(target_arch = "wasm32")) },
    }
}
