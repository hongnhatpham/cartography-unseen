#version 330

uniform sampler2D image_texture;
uniform vec2 uv_scale;
uniform vec2 uv_offset;
uniform float sharpen;

in vec2 uv;
out vec4 frag_color;

void main() {
    vec2 sample_uv = uv * uv_scale + uv_offset;
    vec4 center = texture(image_texture, sample_uv);
    if (sharpen <= 0.0) {
        frag_color = center;
        return;
    }
    vec2 texel = 1.0 / vec2(textureSize(image_texture, 0));
    vec4 blur = (
        texture(image_texture, sample_uv + vec2(texel.x, 0.0)) +
        texture(image_texture, sample_uv - vec2(texel.x, 0.0)) +
        texture(image_texture, sample_uv + vec2(0.0, texel.y)) +
        texture(image_texture, sample_uv - vec2(0.0, texel.y))
    ) * 0.25;
    frag_color = vec4(clamp(center.rgb + (center.rgb - blur.rgb) * sharpen, 0.0, 1.0), center.a);
}
