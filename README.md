# My Blog

A personal blog built with Hugo and deployed on Vercel.

## Local Development

To run the site locally:

```bash
hugo server -D
```

## Deployment

To build and deploy the site to Vercel:

```bash
hugo --minify && vercel deploy --prebuilt
```

To deploy to production:

```bash
hugo --minify && vercel deploy --prebuilt --prod
```

## Project Structure

- `content/`: Contains all blog posts and pages
- `themes/typo/`: The Hugo theme
- `static/`: Static assets
- `layouts/`: Custom layout templates
- `public/`: Generated site (after building)

## Adding New Content

To create a new blog post:

```bash
hugo new posts/category-name/post-name/index.md
```

Add images to the same folder as the index.md file. 