#version 330
uniform sampler2D image_texture;
uniform float opacity;
in vec2 uv;
out vec4 frag_color;
void main() {
    vec4 text = texture(image_texture, uv);
    frag_color = vec4(text.rgb, text.a * opacity);
}
