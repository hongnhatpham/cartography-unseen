#version 330

in vec3 view_dir;

uniform vec3 fog_color;
uniform vec3 zenith_color;
uniform vec3 nadir_color;

out vec4 frag_color;

// Background as a function of look direction: pale fog at the level of the eye,
// rising to a saturated zenith and falling to near black below. proxy.frag runs
// the same function per fragment so distant forms fade into whatever sits
// behind them. The volume has no ground and no horizon line, so this gradient
// is the only thing that tells the sampler which way is up. It must stay in
// sync with the copy in proxy.frag.
vec3 background_color(vec3 dir) {
    float height = clamp(normalize(dir).y, -1.0, 1.0);
    if (height >= 0.0) {
        return mix(fog_color, zenith_color, pow(height, 0.80));
    }
    return mix(fog_color, nadir_color, pow(-height, 1.40));
}

void main() {
    frag_color = vec4(background_color(view_dir), 1.0);
}
