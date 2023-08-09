---
title: "github + cloudflare = <3"
date: 2023-07-22T01:20:00-07:00
draft: true
type: "page"
card_background: "#fe7636"
---

# Motivation
This is a short detail of my journey for trying to host this blog on a custom domain. I was primarily motivated by this because:

- Got tired of _letsencrypt_ expiring every 3 months.
- _GoDaddy_ does not allow automated ssl renewal with letsencrypt.
- Github does not allow custom domains at repo level (I still wanted my toplevel domain to point to my server).
- It's time to move this to JAM stack and say goodbye to my custom built mini blog cms.

# The How
So you want to host a static site on cloudflare with a custom domain with all the goodness of "continuous deployment". Well here is the rundown.

- [Get your markup/markdown content on github/gitlab](#get-content-on-github). I'll cover github here.
- [Setup a build step](#setup-automated-build) if needed in your CI pipeline. I'll cover github actions here.
- Configure cloudflare pages to pick up and host content from the build output.

## Get content on github

- Here I use hugo for this blog. Feel free to use a framework of your choice or _try to_ roll your own!
- Once you have your content uploaded, time to setup a deployment branch. We will store the static generated site in this branch. Run the following command to create a deployment branch.
```bash
git checkout --orphan deploy-pages
git push origin deploy-pages
git checkout main
```

## Setup automated build

- We will use github actions for our build step.

