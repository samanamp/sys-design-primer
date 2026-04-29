// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide'

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			plugins: [starlightThemeRapide()],
			title: 'My Primers',
			// customCss: ['./src/styles/custom.css'],
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/samanamp/sys-design-primer' }],
			sidebar: [
				{
					label: 'LLM Primers',
					// items: [
					// 	// Each item here is one entry in the navigation menu.
					// 	{ label: 'Long-Context LLMs & Context Management', slug: 'primers/1-longcontext' },
					// ],
					autogenerate: { directory: 'llm-primers' },
				},
				{
					label: 'LLM Sys Design',
					autogenerate: { directory: 'llm-sysdesign' },
				},
				{
					label: 'Primers',
					autogenerate: { directory: 'primers' },
				},
				{
					label: 'Sys Design',
					autogenerate: { directory: 'sysdesign' },
				},
				{
					label: 'Misc',
					autogenerate: { directory: 'misc' },
				},
				{
					label: 'Modern Coding',
					autogenerate: { directory: 'coding' },
				},
				{
					label: '40min interviews',
					autogenerate: { directory: '40-min-interview' },
				},
			],
		}),
	],
});
