// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide'
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	devToolbar: { enabled: false },
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			plugins: [starlightThemeRapide()],
			title: 'My Primers',
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'link',
					attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
				},
				{
					tag: 'link',
					attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
				},
				{
					tag: 'script',
					attrs: {
						src: '/scripts/hide-header-on-scroll.js',
						defer: true,
					},
				},
				{
					tag: 'script',
					attrs: {
						src: '/scripts/zen-mode.js',
						defer: true,
					},
				},
				{
					tag: 'script',
					attrs: {
						src: '/scripts/fullscreen.js',
						defer: true,
					},
				},
			],
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
					label: 'Diffusion',
					autogenerate: { directory: 'diffusion' },
				},
				{
					label: 'ML Design',
					autogenerate: { directory: 'ml-design' },
				},
				{
					label: 'ML Breadth',
					autogenerate: { directory: 'ml-breadth' },
				},
				{
					label: 'ML Modeling Fundamentals',
					autogenerate: { directory: 'ml-modeling-fundamentals' },
				},
				{
					label: 'Optimization',
					autogenerate: { directory: 'optimization' },
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
					label: 'Linear Algebra & Backprop',
					autogenerate: { directory: 'ml-coding' },
				},
				{
					label: 'Modern Coding',
					autogenerate: { directory: 'coding' },
				},
				{
					label: '40min interviews',
					autogenerate: { directory: '40-min-interview' },
				},
				{
					label: 'prachub',
					autogenerate: { directory: 'prachub-designs' },
				},
				{
					label: 'Paper Mocks',
					autogenerate: { directory: 'papers' },
				},
			],
		}),
	],
});
