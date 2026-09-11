import { afterEach, expect, it, vi } from 'vitest';
import { exportJWK, generateKeyPair, SignJWT } from 'jose';
import { verifyHuman } from '../worker/auth';

afterEach(() => vi.unstubAllGlobals());

it('verifies Access signatures, issuer, audience, expiry and human identity', async () => {
  const { privateKey, publicKey } = await generateKeyPair('RS256');
  const jwk = await exportJWK(publicKey);
  const fetchKeys = vi.fn(async () => new Response(JSON.stringify({ keys: [{ ...jwk, kid: 'unit-key', alg: 'RS256', use: 'sig' }] }), { headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetchKeys);
  const env = { ACCESS_TEAM_DOMAIN: 'signed-test.cloudflareaccess.com', ACCESS_AUD: 'human-dashboard' };
  const sign = (claims: Record<string, unknown> = {}, key = privateKey) => new SignJWT({ email: 'test@example.test', ...claims })
    .setProtectedHeader({ alg: 'RS256', kid: 'unit-key' })
    .setSubject('user-123').setIssuedAt()
    .setIssuer(typeof claims.iss === 'string' ? claims.iss : 'https://signed-test.cloudflareaccess.com')
    .setAudience(typeof claims.aud === 'string' ? claims.aud : 'human-dashboard')
    .setExpirationTime(typeof claims.exp === 'number' ? claims.exp : '5m').sign(key);
  const request = (token: string) => new Request('https://monitor.test/', { headers: { 'Cf-Access-Jwt-Assertion': token } });
  expect(await verifyHuman(request(await sign()), env)).toBe('user-123');
  expect(fetchKeys).toHaveBeenCalledOnce();
  for (const claims of [{ aud: 'another-app' }, { iss: 'https://attacker.test' }, { exp: 1 }, { email: null }]) {
    await expect(verifyHuman(request(await sign(claims)), env)).rejects.toThrow();
  }
  const otherKey = (await generateKeyPair('RS256')).privateKey;
  await expect(verifyHuman(request(await sign({}, otherKey)), env)).rejects.toThrow();
});
