using OptimaSolver
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(
    OptimaSolver,
    :DocTestSetup,
    :(using OptimaSolver);
    recursive = true,
)

# ── Stopgap: heading anchors that contain LaTeX ──────────────────────────────
# DocumenterVitepress builds each heading as `## <text> {#<slug>}`, where the
# slug is Documenter's anchor label passed through its own
# `sanitized_anchor_label` — whose comment says "vitepress doesn't like special
# markdown characters in the id slug", but which only strips `[ ] ( ) *`.
#
# A heading such as ``## The Schur complement ``\Sigma``` yields a slug holding
# a backslash and braces. VitePress's `{#...}` parser rejects them, so it treats
# the whole suffix as *text*: the heading renders with the raw `{#...}` visible,
# the formula is dropped, and the same garbage lands in the "On this page"
# outline. This matters here more than anywhere else in the ecosystem —
# `theory.md` alone carries most of this package's 310 formulas.
#
# Stripping those characters from the slug is safe: this narrows to headings
# only, leaving docstring anchors — which legitimately carry braces, are emitted
# as raw `<a id=…>`, and *are* linked to — untouched.
#
# Remove once `sanitized_anchor_label` covers these characters upstream.
function DocumenterVitepress.render(
        io::IO,
        mime::MIME"text/plain",
        node::Documenter.MarkdownAST.Node,
        header::Documenter.AnchoredHeader,
        page,
        doc;
        kwargs...,
    )
    anchor = header.anchor
    label = DocumenterVitepress.sanitized_anchor_label(anchor)
    id = replace(replace(label, r"[\\{}]" => ""), " " => "-")
    heading = first(node.children)
    println(io)
    print(io, "#"^(heading.element.level), " ")
    heading_iob = IOBuffer()
    DocumenterVitepress.render(heading_iob, mime, node, heading.children, page, doc; kwargs...)
    print(io, rstrip(String(take!(heading_iob))))
    print(io, " {#$(id)}")
    if haskey(kwargs, :inventory)
        item = DocumenterVitepress.InventoryItem(
            name = anchor.id,
            domain = "std",
            role = "label",
            dispname = DocumenterVitepress._get_inventory_dispname(
                anchor.id, Documenter.MDFlatten.mdflatten(anchor.node)
            ),
            priority = -1,
            uri = DocumenterVitepress._get_inventory_uri(doc, page, id),
        )
        push!(kwargs[:inventory], item)
    end
    println(io)
    return nothing
end

# ── Stopgap: ordered lists start at 2, and swallow their first item ──────────
# DocumenterVitepress numbers ordered-list items with `bullet(i) = "$(i+1). "`,
# but `enumerate` is already 1-based, so every ordered list comes out numbered
# from 2. It also emits no blank line before the list.
#
# Together those two do real damage: a list whose first marker is `2.` cannot
# interrupt a paragraph — CommonMark allows that only for a list starting at
# `1.` — so with no separating blank line the first item is absorbed into the
# preceding prose as plain text and the list begins at `3.`.
#
# Remove once the numbering is fixed upstream.
function DocumenterVitepress.render(
        io::IO,
        mime::MIME"text/plain",
        node::Documenter.MarkdownAST.Node,
        list::Documenter.MarkdownAST.List,
        page,
        doc;
        kwargs...,
    )
    bullet(i) = list.type === :ordered ? "$(i). " : "- "
    println(io)
    iob = IOBuffer()
    for (i, item) in enumerate(node.children)
        DocumenterVitepress.render(
            iob, mime, item, item.children, page, doc; prenewline = false, kwargs...
        )
        eachline = split(String(take!(iob)), '\n')
        # Continuation lines must line up under the marker's full width. Upstream
        # hard-codes two spaces, which fits `- ` but not `1. `: a display equation
        # inside an ordered item falls out of the list, splitting it in two.
        pad = " "^length(bullet(i))
        eachline[2:end] .= pad .* eachline[2:end]
        final_string = join(eachline, '\n')
        endswith(final_string, '\n') || (final_string *= "\n")
        print(io, bullet(i))
        print(io, final_string)
    end
    return nothing
end

makedocs(;
    # `clean = false` let pages deleted from the source survive in `build/` and
    # go on being deployed. Nothing writes into `build/` before `makedocs`, so
    # wiping it costs nothing.
    clean = true,
    modules = [OptimaSolver],
    authors = "Jean-François Barthélémy",
    sitename = "OptimaSolver.jl",
    remotes = nothing,
    # The favicon and the logo are picked up automatically from `docs/src/assets`,
    # and the sidebar is derived from `pages` below, so neither needs declaring.
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl",
        description = "Primal-dual interior-point solver for Gibbs energy minimization in Julia",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory" => "theory.md",
        "Examples" => [
            "Basic Usage" => "examples/basic_usage.md",
            "Warm Start" => "examples/warm_start.md",
            "Sensitivity" => "examples/sensitivity.md",
            "SciML Interface" => "examples/sciml_interface.md",
        ],
        "API Reference" => "api.md",
    ],
    # Only exported names have to appear on a curated page; the internals are
    # documented for the reader of the source. Same setting as TensND and MFH.
    checkdocs = :exports,
    warnonly = [:missing_docs, :docs_block],
)

# DocumenterVitepress writes a real directory per version rather than the
# symlinks Documenter used, so it needs its own `deploydocs`.
DocumenterVitepress.deploydocs(;
    repo = "github.com/MicroPoroChemoMechanics/OptimaSolver.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = false,
)
