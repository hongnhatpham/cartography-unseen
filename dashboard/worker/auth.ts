import { createRemoteJWKSet, jwtVerify } from 'jose';

export interface AccessConfig { ACCESS_TEAM_DOMAIN: string; ACCESS_AUD: string }
export type HumanVerifier = (request: Request, env: AccessConfig) => Promise<string>;
const keySets = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

export const verifyHuman: HumanVerifier = async (request, env) => {
  if (!/^[a-z0-9-]+\.cloudflareaccess\.com$/.test(env.ACCESS_TEAM_DOMAIN) || !env.ACCESS_AUD || env.ACCESS_AUD === 'configure-before-deployment') {
    throw new Error('Access configuration missing');
  }
  const token = request.headers.get('Cf-Access-Jwt-Assertion');
  if (!token || token.length > 8192) throw new Error('Access assertion missing');
  const issuer = `https://${env.ACCESS_TEAM_DOMAIN}`;
  let keys = keySets.get(issuer);
  if (!keys) {
    keys = createRemoteJWKSet(new URL(`${issuer}/cdn-cgi/access/certs`));
    keySets.set(issuer, keys);
  }
  const { payload } = await jwtVerify(token, keys, {
    issuer, audience: env.ACCESS_AUD, algorithms: ['RS256'],
    requiredClaims: ['exp', 'iat', 'sub'],
  });
  // Service-token JWTs have no user email and cannot read the human dashboard.
  if (typeof payload.email !== 'string' || !payload.email || !payload.sub) throw new Error('Human identity required');
  return payload.sub;
};

export async function tokenHash(token: string): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token));
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
}
