#version 330

in vec2 uv;

uniform vec3 fog_color;
uniform vec3 zenith_color;
uniform float horizon_v;

out vec4 frag_color;

// Sky gradient behind the landscape: fog colour at the horizon rising to a
// darker, more saturated zenith. A flat white sky above eye level was read by
// the sampler as a ceiling or a window wall; a graded sky reads as outdoors.
void main() {
    float t = clamp((uv.y - horizon_v) / max(1.0 - horizon_v, 0.05), 0.0, 1.0);
    t = pow(t, 0.8);
    vec3 color = mix(fog_color, zenith_color, t);
    frag_color = vec4(color, 1.0);
}
