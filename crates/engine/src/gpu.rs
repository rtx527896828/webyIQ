//! Small wgpu helpers: bind layouts, pipelines, shader variants.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Bind {
    Read,
    ReadWrite,
    Uniform,
}

pub(crate) const MAX_WG_X: u32 = 65535;

pub(crate) fn bind_layout(device: &wgpu::Device, kinds: &[Bind]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match k {
                Bind::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Bind::Read | Bind::ReadWrite => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: matches!(k, Bind::Read) },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
}

pub(crate) fn bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

pub(crate) fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    label: &str,
    entry: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
        cache: None,
    })
}

/// WGSL fallbacks for the `packed_4x8_integer_dot_product` builtins, for browsers without it.
const PACKED_INT8_POLYFILL: &str = r#"
fn dot4I8_poly(a: u32, b: u32) -> i32 {
    let va = vec4<i32>(bitcast<i32>(a << 24u), bitcast<i32>(a << 16u), bitcast<i32>(a << 8u), bitcast<i32>(a)) >> vec4<u32>(24u);
    let vb = vec4<i32>(bitcast<i32>(b << 24u), bitcast<i32>(b << 16u), bitcast<i32>(b << 8u), bitcast<i32>(b)) >> vec4<u32>(24u);
    return dot(va, vb);
}

fn pack4xI8_poly(v: vec4<i32>) -> u32 {
    let u = vec4<u32>(bitcast<vec4<u32>>(v) & vec4<u32>(0xFFu)) << vec4<u32>(0u, 8u, 16u, 24u);
    return u.x | u.y | u.z | u.w;
}
"#;

/// Device capabilities that select shader variants.
#[derive(Clone, Copy, Debug)]
pub struct ShaderCaps {
    /// WGSL `packed_4x8_integer_dot_product` (always natively; `navigator.gpu.wgslLanguageFeatures` on the web)
    pub packed_int8: bool,
    /// `Features::SUBGROUP` enabled on the device
    pub subgroups: bool,
}

/// Resolve `//#if SUBGROUPS` / `//#else` / `//#endif` blocks and builtin polyfills.
pub(crate) fn shader_source(src: &str, caps: ShaderCaps) -> String {
    let mut out = String::new();
    let mut keep = vec![true];
    for line in src.lines() {
        match line.trim() {
            "//#if SUBGROUPS" => keep.push(caps.subgroups),
            "//#else" => {
                let v = keep.pop().unwrap();
                keep.push(!v);
            }
            "//#endif" => {
                keep.pop();
            }
            _ if keep.iter().all(|&k| k) => {
                out.push_str(line);
                out.push('\n');
            }
            _ => {}
        }
    }
    // browsers (Tint) require the enable directive; naga rejects it and allows subgroup builtins without it
    if cfg!(target_arch = "wasm32") && caps.subgroups && out.contains("subgroupAdd") {
        out.insert_str(0, "enable subgroups;\n");
    }
    if !caps.packed_int8 {
        out = out.replace("dot4I8Packed(", "dot4I8_poly(").replace("pack4xI8(", "pack4xI8_poly(");
        out.push_str(PACKED_INT8_POLYFILL);
    }
    out
}

/// Split a 1D workgroup count into (x, y) so x stays under the per-dimension limit.
pub(crate) fn grid(n: u32) -> (u32, u32) {
    if n <= MAX_WG_X { (n, 1) } else { (MAX_WG_X, n.div_ceil(MAX_WG_X)) }
}
