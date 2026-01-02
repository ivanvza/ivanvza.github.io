import { QuartzTransformerPlugin } from "../types"
import { FilePath, slugifyFilePath } from "../../util/path"
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
  slug: string
  filePath: string
  title: string
  date: Date | null
  description: string
  tags: string[]
  draft: boolean
}

function getAllMarkdownFiles(dir: string, baseDir: string = dir): string[] {
  const files: string[] = []

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true })

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name)

      if (entry.isDirectory()) {
        // Skip hidden directories and common ignore patterns
        if (!entry.name.startsWith(".") && entry.name !== "templates" && entry.name !== "private") {
          files.push(...getAllMarkdownFiles(fullPath, baseDir))
        }
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        // Get relative path from content dir
        const relativePath = path.relative(baseDir, fullPath)
        files.push(relativePath)
      }
    }
  } catch {
    // Directory doesn't exist or can't be read
  }

  return files
}

function parsePageInfo(filePath: string, contentDir: string): PageInfo | null {
  const fullPath = path.join(contentDir, filePath)

  try {
    const content = fs.readFileSync(fullPath, "utf-8")
    const { data, content: body } = matter(content)

    // Generate slug from file path (same way Quartz does it)
    const slug = slugifyFilePath(filePath as FilePath)

    // Extract title (from frontmatter or filename)
    const filename = path.basename(filePath, ".md")
    const title = data.title || filename

    // Extract date
    let date: Date | null = null
    if (data.date) {
      date = new Date(data.date)
    }

    // Check if draft
    const draft = data.draft === true

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
          !trimmed.startsWith(">") &&
          !trimmed.startsWith("|") &&
          !trimmed.startsWith("-")
        )
      })
      if (lines.length > 0) {
        description = lines[0].trim()
      }
    }

    // Extract tags
    const tags: string[] = data.tags || []

    return { slug, filePath, title, date, description, tags, draft }
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

      // argv.directory is the content folder (e.g., "content")
      const contentDir = ctx.argv.directory

      // Get all markdown files
      const allFiles = getAllMarkdownFiles(contentDir)

      // Parse info for each page
      const pages: PageInfo[] = []
      for (const filePath of allFiles) {
        const info = parsePageInfo(filePath, contentDir)
        if (info) {
          // Skip excluded slugs, drafts, and tag pages
          const isExcluded =
            opts.exclude.includes(info.slug) ||
            info.slug.startsWith("tags/") ||
            info.slug.endsWith("/index") ||
            info.draft

          if (!isExcluded) {
            pages.push(info)
          }
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

        // Title as wiki-link (use filename for link, title for display)
        const linkTarget = page.filePath.replace(/\.md$/, "")
        lines.push(`### [[${linkTarget}|${page.title}]]`)

        // Date and description on same line
        const meta: string[] = []
        if (opts.showDate && page.date) {
          meta.push(`*${formatDate(page.date)}*`)
        }
        if (opts.showDescription && page.description) {
          const desc = truncate(page.description, opts.descriptionLength)
          if (meta.length > 0) {
            meta.push(` — ${desc}`)
          } else {
            meta.push(desc)
          }
        }
        if (meta.length > 0) {
          lines.push(meta.join(""))
        }

        // Tags as clickable links
        if (opts.showTags && page.tags.length > 0) {
          const tagLinks = page.tags.map((tag) => `[[tags/${tag}|#${tag}]]`).join(" ")
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
