import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { FullSlug, resolveRelative, simplifySlug } from "../util/path"
import { QuartzPluginData } from "../plugins/vfile"
import { byDateAndAlphabetical } from "./PageList"
import { Date, getDate } from "./Date"
import { classNames } from "../util/lang"

interface Options {
  title?: string
  showTags: boolean
  showDescription: boolean
  descriptionLength: number
}

const defaultOptions: Options = {
  showTags: true,
  showDescription: true,
  descriptionLength: 150,
}

function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength).trim() + "..."
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const AllPosts: QuartzComponent = ({
    allFiles,
    fileData,
    displayClass,
    cfg,
  }: QuartzComponentProps) => {
    // Only render on the index page
    const currentSlug = simplifySlug(fileData.slug!)
    if (currentSlug !== "/") {
      return null
    }

    // Filter and sort pages
    const pages = allFiles
      .filter((file) => {
        const slug = simplifySlug(file.slug!)
        // Exclude index, 404, tags, and folder index pages
        return (
          slug !== "/" &&
          !file.slug?.startsWith("tags/") &&
          !file.slug?.endsWith("/index") &&
          file.slug !== "404"
        )
      })
      .sort(byDateAndAlphabetical(cfg))

    return (
      <div class={classNames(displayClass, "all-posts")}>
        {opts.title && <h2>{opts.title}</h2>}
        <ul class="all-posts-ul">
          {pages.map((page) => {
            const title = page.frontmatter?.title ?? page.slug
            const tags = page.frontmatter?.tags ?? []
            const description = page.description
              ? truncate(page.description, opts.descriptionLength)
              : ""

            return (
              <li class="all-posts-li">
                <div class="all-posts-item">
                  <h3>
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class="internal">
                      {title}
                    </a>
                  </h3>
                  <div class="meta-container">
                    {page.dates && (
                      <span class="meta">
                        <Date date={getDate(cfg, page)!} locale={cfg.locale} />
                      </span>
                    )}
                    {opts.showDescription && description && (
                      <span class="description"> — {description}</span>
                    )}
                  </div>
                  {opts.showTags && tags.length > 0 && (
                    <ul class="tags">
                      {tags.map((tag) => (
                        <li>
                          <a
                            class="internal tag-link"
                            href={resolveRelative(fileData.slug!, `tags/${tag}` as FullSlug)}
                          >
                            {tag}
                          </a>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      </div>
    )
  }

  AllPosts.css = `
.all-posts-ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.all-posts-li {
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--lightgray);
}

.all-posts-li:last-child {
  border-bottom: none;
}

.all-posts-item h3 {
  margin: 0 0 0.5rem 0;
}

.all-posts-item h3 a {
  font-weight: 600;
}

.all-posts-item .meta-container {
  color: var(--gray);
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.all-posts-item .meta {
  font-style: italic;
}

.all-posts-item .description {
  color: var(--darkgray);
}

.all-posts-item .tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  list-style: none;
  padding: 0;
  margin: 0;
}

.all-posts-item .tags li {
  margin: 0;
}

.all-posts-item .tag-link {
  font-size: 0.85rem;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  background: var(--highlight);
}
`

  return AllPosts
}) satisfies QuartzComponentConstructor
