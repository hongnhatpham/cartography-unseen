#version 330

uniform sampler2D source_image;
uniform sampler2D source_depth;
uniform sampler2D current_depth;
uniform mat4 current_inverse_view_projection;
uniform mat4 source_view_projection;
uniform vec2 fallback_uv_scale;
uniform vec2 fallback_uv_offset;
uniform float warp_strength;
uniform float occlusion_tolerance;

in vec2 uv;
out vec4 frag_color;

void main() {
    vec2 fallback_uv = clamp(
        uv * fallback_uv_scale + fallback_uv_offset,
        vec2(0.001),
        vec2(0.999)
    );
    vec3 fallback = texture(source_image, fallback_uv).rgb;
    float depth = texture(current_depth, uv).r;

    // The far plane has no reconstructable proxy geometry (normally sky), so
    // use the smooth camera approximation rather than creating an empty hole.
    if (depth >= 0.9998) {
        frag_color = vec4(fallback, 1.0);
        return;
    }

    vec4 current_clip = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = current_inverse_view_projection * current_clip;
    if (abs(world.w) < 1e-6) {
        frag_color = vec4(fallback, 1.0);
        return;
    }
    world /= world.w;

    vec4 source_clip = source_view_projection * world;
    if (source_clip.w <= 1e-6) {
        frag_color = vec4(fallback, 1.0);
        return;
    }
    vec3 source_ndc = source_clip.xyz / source_clip.w;
    vec2 geometric_uv = source_ndc.xy * 0.5 + 0.5;
    bool in_source = all(greaterThanEqual(geometric_uv, vec2(0.001))) &&
                     all(lessThanEqual(geometric_uv, vec2(0.999)));
    if (!in_source) {
        frag_color = vec4(fallback, 1.0);
        return;
    }

    // A partial warp keeps movement responsive without pulling the old frame
    // all the way to the new camera pose. Occluded/disoccluded pixels are
    // rejected against the source depth and fade to the smooth fallback.
    vec2 sample_uv = mix(uv, geometric_uv, warp_strength);
    float expected_depth = source_ndc.z * 0.5 + 0.5;
    float observed_depth = texture(source_depth, geometric_uv).r;
    float behind_source = max(expected_depth - observed_depth, 0.0);
    float confidence = 1.0 - smoothstep(
        occlusion_tolerance,
        occlusion_tolerance * 4.0,
        behind_source
    );
    float depth_edge = fwidth(observed_depth);
    confidence *= 1.0 - smoothstep(0.0008, 0.006, depth_edge);

    vec3 warped = texture(source_image, clamp(sample_uv, vec2(0.001), vec2(0.999))).rgb;
    frag_color = vec4(mix(fallback, warped, confidence), 1.0);
}
