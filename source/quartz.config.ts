import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "0xIvan",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: { provider: 'umami', host: 'ivanvza.github.io', websiteId: 'be56f9f3-15a0-4ab9-a2f6-781e2ebc465d' },
    locale: "en-US",
    baseUrl: "ivanvza.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#ffffff",
          lightgray: "#f4f4f5",
          gray: "#a1a1aa",
          darkgray: "#27272a",
          dark: "#09090b",
          secondary: "#2563eb",
          tertiary: "#ea580c",
          highlight: "rgba(37, 99, 235, 0.1)",
          textHighlight: "#3b82f644",
        },
        darkMode: {
          light: "#09090b",
          lightgray: "#18181b",
          gray: "#3f3f46",
          darkgray: "#e4e4e7",
          dark: "#fafafa",
          secondary: "#3b82f6",
          tertiary: "#f97316",
          highlight: "rgba(59, 130, 246, 0.1)",
          textHighlight: "#3b82f633",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
