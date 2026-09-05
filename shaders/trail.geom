#version 330

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 window_size;
in float vertex_opacity[];
out float opacity;

void corner(vec4 position, vec2 offset, float alpha) {
    gl_Position = position + vec4(offset * position.w, 0.0, 0.0);
    opacity = alpha;
    EmitVertex();
}

void main() {
    vec4 a = gl_in[0].gl_Position;
    vec4 b = gl_in[1].gl_Position;
    float alpha_a = vertex_opacity[0];
    float alpha_b = vertex_opacity[1];
    // Clip before perspective division: a segment crossing the camera must
    // end at the near plane instead of flipping or expanding across the view.
    float near_a = a.z + a.w;
    float near_b = b.z + b.w;
    if (near_a < 0.0 && near_b < 0.0) return;
    if (near_a < 0.0 || near_b < 0.0) {
        float t = near_a / (near_a - near_b);
        vec4 clipped = mix(a, b, t);
        float alpha = mix(alpha_a, alpha_b, t);
        if (near_a < 0.0) { a = clipped; alpha_a = alpha; }
        else { b = clipped; alpha_b = alpha; }
    }
    if (a.w <= 0.0 || b.w <= 0.0) return;
    vec2 direction = (b.xy / b.w - a.xy / a.w) * window_size;
    float length_px = length(direction);
    if (length_px < 0.00001) return;
    // Fifteen pixels regardless of the driver's native GL line-width limit.
    vec2 offset = vec2(-direction.y, direction.x) / length_px * 15.0 / window_size;
    corner(a, offset, alpha_a);
    corner(a, -offset, alpha_a);
    corner(b, offset, alpha_b);
    corner(b, -offset, alpha_b);
    EndPrimitive();
}
