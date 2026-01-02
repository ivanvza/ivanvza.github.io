import { QuartzTransformerPlugin } from "../types"
import { simplifySlug, FullSlug } from "../../util/path"
import fs from "fs"
import path from "path"
import matter from "gray-matter"

export interface Options {
  /** The marker to replace with wiki-links. Default: <!-- all-pages --> */
  marker: string
  /** Exclude these slugs from the generated links */
  exclude: string[]
  /** Title for the links section */
  title: string
  /** Show description under each link */
  showDescription: boolean
  /** Show date for each link */
  showDate: boolean
  /** Show tags for each link */
  showTags: boolean
  /** Maximum description length */
  descriptionLength: number
}

const defaultOptions: Options = {
  marker: "<!-- all-pages -->",
  exclude: ["index", "404"],
  title: "",
  showDescription: true,
  showDate: true,
  showTags: true,
  descriptionLength: 150,
}

interface PageInfo {
  slug: FullSlug
  title: string
  date: Date | null
  description: string
  tags: string[]
}

function parsePageInfo(slug: FullSlug, contentDir: string): PageInfo | null {
  // Convert slug to file path
  const filePath = path.join(contentDir, slug + ".md")

  try {
    if (!fs.existsSync(filePath)) {
      return null
    }

    const content = fs.readFileSync(filePath, "utf-8")
    const { data, content: body } = matter(content)

    // Extract title (from frontmatter or slug)
    const title = data.title || slug.split("/").pop() || slug

    // Extract date
    let date: Date | null = null
    if (data.date) {
      date = new Date(data.date)
    }

    // Extract description (from frontmatter or first paragraph)
    let description = ""
    if (data.description) {
      description = data.description
    } else {
      // Get first non-empty paragraph from content
      const lines = body.split("\n").filter((line) => {
        const trimmed = line.trim()
        return (
          trimmed.length > 0 &&
          !trimmed.startsWith("#") &&
          !trimmed.startsWith("---") &&
          !trimmed.startsWith("```") &&
          !trimmed.startsWith("<!--") &&
          !trimmed.startsWith("![") &&
          !trimmed.startsWith(">")
        )
      })
      if (lines.length > 0) {
        description = lines[0].trim()
      }
    }

    // Extract tags
    const tags: string[] = data.tags || []

    return { slug, title, date, description, tags }
  } catch {
    return null
  }
}

function formatDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength).trim() + "..."
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

      // Get content directory from config
      const contentDir = path.join(ctx.argv.directory, "content")

      // Get all page slugs except excluded ones
      const allSlugs = ctx.allSlugs.filter((slug) => {
        const simple = simplifySlug(slug)
        return (
          !opts.exclude.includes(slug) &&
          !slug.startsWith("tags/") &&
          !slug.endsWith("/index")
        )
      })

      // Parse info for each page
      const pages: PageInfo[] = []
      for (const slug of allSlugs) {
        const info = parsePageInfo(slug, contentDir)
        if (info) {
          pages.push(info)
        }
      }

      // Sort by date (newest first), pages without dates go to the end
      pages.sort((a, b) => {
        if (a.date && b.date) {
          return b.date.getTime() - a.date.getTime()
        }
        if (a.date) return -1
        if (b.date) return 1
        return a.title.localeCompare(b.title)
      })

      // Generate markdown for each page
      const entries = pages.map((page) => {
        const lines: string[] = []

        // Title as wiki-link
        lines.push(`### [[${page.slug}|${page.title}]]`)

        // Date and description on same line
        const meta: string[] = []
        if (opts.showDate && page.date) {
          meta.push(`*${formatDate(page.date)}*`)
        }
        if (opts.showDescription && page.description) {
          const desc = truncate(page.description, opts.descriptionLength)
          if (meta.length > 0) {
            meta.push(` · ${desc}`)
          } else {
            meta.push(desc)
          }
        }
        if (meta.length > 0) {
          lines.push(meta.join(""))
        }

        // Tags
        if (opts.showTags && page.tags.length > 0) {
          const tagLinks = page.tags.map((tag) => `#${tag}`).join(" ")
          lines.push(tagLinks)
        }

        return lines.join("\n")
      })

      // Build the replacement content
      let replacement = ""
      if (opts.title) {
        replacement += `## ${opts.title}\n\n`
      }
      replacement += entries.join("\n\n")

      // Replace the marker with the generated links
      return src.replace(opts.marker, replacement)
    },
  }
}
