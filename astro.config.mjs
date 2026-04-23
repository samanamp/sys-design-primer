// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide'

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			plugins: [starlightThemeRapide()],
			title: 'My Docs',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/withastro/starlight' }],
			sidebar: [
				{
					label: 'LLM Primers',
					// items: [
					// 	// Each item here is one entry in the navigation menu.
					// 	{ label: 'Long-Context LLMs & Context Management', slug: 'primers/1-longcontext' },
					// ],
					autogenerate: { directory: 'primers' },
				},
				{
					label: 'LLM Sys Design',
					autogenerate: { directory: 'llm-sysdesign' },
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
