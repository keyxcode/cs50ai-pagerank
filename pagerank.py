import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    corpus_copy = corpus.copy()

    model = {page: 0 for page in corpus_copy}
    num_page = len(model)

    # if page has no outgoing links, pretend it has links to all pages in the corpus, including itself
    if len(corpus_copy[page]) == 0:
        corpus_copy[page] = {page for page in corpus_copy}

    links = corpus_copy[page]
    num_links = len(links)

    # with probability damping_factor, choose one of the links with equal probability
    link_probability = round(damping_factor / num_links, 5)
    for link in links:
        model[link] = link_probability

    # with probability (1 - damping_factor), randomly choose one of all pages in the corpus with equal probability
    page_probability = round((1 - damping_factor) / num_page, 5)
    for page in corpus_copy:
        model[page] += page_probability

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page = random.choice(list(corpus.keys()))
    samples = [page]

    for _ in range(n - 1):
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), weights=model.values())[0]
        samples.append(page)

    ranks = {page: samples.count(page) / n for page in samples}

    return ranks


def is_coverged(old_ranks, new_ranks):
    if old_ranks == {} or new_ranks == {}:
        return False

    CONVERGENCE = 0.001

    for page in old_ranks:
        if new_ranks[page] - old_ranks[page] > CONVERGENCE:
            return False

    return True


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # assigning each page a rank of 1 / total number of pages in the corpus
    corpus_copy = corpus.copy()
    num_pages = len(corpus_copy)
    equal_probability = 1 / num_pages

    # initialize page ranks
    ranks = {page: equal_probability for page in list(corpus_copy.keys())}
    old_ranks = dict()

    while True:
        if is_coverged(old_ranks, ranks):
            break

        old_ranks = ranks.copy()

        for rank in ranks:
            random_prob = (1 - damping_factor) / num_pages
            link_prob = 0
            # page and rank represent the same thing in two different contexts: a page name string
            for page in corpus_copy:
                # a page that has no links at all is interpreted as having one link for every page in the corpus
                if len(corpus_copy[page]) == 0:
                    corpus_copy[page] = {page for page in corpus_copy}
                # a page that has link to the current rank
                if rank in corpus_copy[page]:
                    link_prob += ranks[page] / len(corpus_copy[page])

            ranks[rank] = random_prob + damping_factor * link_prob

    return ranks


if __name__ == "__main__":
    main()
