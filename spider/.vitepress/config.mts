import baseConfig from '../docs/.vitepress/config.mts';

// Allow running `vitepress build` from the repository root by pointing to the docs directory.
export default {
  ...baseConfig,
  srcDir: 'docs',
  outDir: 'docs/.vitepress/dist',
  tempDir: 'docs/.vitepress/.temp',
  cacheDir: 'docs/.vitepress/cache',
};
