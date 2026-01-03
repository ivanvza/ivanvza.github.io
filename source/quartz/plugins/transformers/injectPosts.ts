import { QuartzTransformerPlugin } from "../types"
import { FullSlug } from "../../util/path"

interface Options {
  marker: string
  excludePatterns: string[]
}

const defaultOptions: Options = {
  marker: "<!-- INJECT_ALL_POSTS -->",
  excludePatterns: ["index", "404", "tags/"],
}

export const InjectAllPosts: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "InjectAllPosts",
    textTransform(ctx, src) {
      const content = src.toString()

      // Only process files containing the marker
      if (!content.includes(opts.marker)) {
        return src
      }

      // Get all post slugs, excluding patterns
      const postSlugs = ctx.allSlugs.filter((slug: FullSlug) => {
        return !opts.excludePatterns.some(
          (pattern) =>
            slug === pattern || slug.startsWith(pattern) || slug.endsWith("/index"),
        )
      })

      // Generate wikilinks for graph connectivity
      // Hidden via CSS - AllPosts handles the nice display
      // Requires enableInHtmlEmbed: true in OFM config
      const wikilinks = postSlugs
        .map((slug) => {
          const title = slug.split("/").pop()!.replace(/-/g, " ")
          return `[[${slug}|${title}]]`
        })
        .join(" ")

      // Wrap in hidden div - wikilinks still get processed for graph
      const injection = `<div class="graph-links-hidden">${wikilinks}</div>`

      return content.replace(opts.marker, injection)
    },
  }
}
