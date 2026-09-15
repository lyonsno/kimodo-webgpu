/**
 * Node module hooks so tests can import the shipped GPU modules, whose
 * shader imports use Vite's `?raw` suffix. Each .wgsl resolves to an empty
 * default-export string — tests exercise dispatch/submission structure, not
 * shader compilation.
 */
import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.includes('.wgsl')) {
      const clean = specifier.split('?')[0];
      const url = new URL(clean, context.parentURL).href + '?raw';
      return { url, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.includes('.wgsl')) {
      return { format: 'module', source: 'export default "";', shortCircuit: true };
    }
    return nextLoad(url, context);
  },
});
