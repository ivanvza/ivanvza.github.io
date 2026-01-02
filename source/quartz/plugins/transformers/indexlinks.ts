import { QuartzTransformerPlugin } from "../types"
import { simplifySlug } from "../../util/path"

export interface Options {
  /** The marker to replace with wiki-links. Default: <!-- all-pages --> */
  marker: string
  /** Exclude these slugs from the generated links */
  exclude: string[]
  /** Title for the links section */
  title: string
}

const defaultOptions: Options = {
  marker: "<!-- all-pages -->",
  exclude: ["index", "404"],
  title: "",
}

export const IndexLinks: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "IndexLinks",
    textTransform(ctx, src) {
      if (src instanceof Buffer) {
        src = src.toString()
      }

      // Only process if the marker exists
      if (!src.includes(opts.marker)) {
        return src
      }

      // Get all page slugs except excluded ones
      const allPages = ctx.allSlugs
        .filter((slug) => {
          const simple = simplifySlug(slug)
          // Exclude specified slugs and tag/folder index pages
          return (
            !opts.exclude.includes(slug) &&
            !slug.startsWith("tags/") &&
            !slug.endsWith("/index")
          )
        })
        .sort()

      // Generate wiki-links
      const links = allPages.map((slug) => `- [[${slug}]]`).join("\n")

      // Build the replacement content
      let replacement = ""
      if (opts.title) {
        replacement += `## ${opts.title}\n\n`
      }
      replacement += links

      // Replace the marker with the generated links
      return src.replace(opts.marker, replacement)
    },
  }
}
