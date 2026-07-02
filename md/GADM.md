# State Boundaries (GADM): What the Data Shows

> Last updated: 2026-07-02

This dataset holds Malaysia's state and federal-territory boundaries — all 16 units
(13 states plus Kuala Lumpur, Labuan, and Putrajaya). It provides the geographic
backbone for connecting states to each other and for placing other data (population,
rainfall) on the map. Each section uses **Main idea**, **Evidence**, **Analysis**,
and **Link**.

---

## Pattern 1 — Peninsular States Are Connected; Borneo Is Separate

**Main idea.** Malaysia splits into two well-defined groups: a densely connected
Peninsula and the isolated East Malaysian states of Sabah and Sarawak.

**Evidence.** Building a network where states are linked if they share a land border
produces one large connected cluster on the Peninsula. Sabah and Sarawak share no
land border with Peninsular states (they are across the South China Sea) and so have
no connections in that network. Pahang is the most connected state (six neighbours),
followed by Perak (five). The federal territories are small enclaves with only one or
two borders each.

**Analysis.** This border structure is exactly the kind of relationship graph-based
models can use. The pipeline saves it both as a simple yes/no connection matrix and
as a version weighted by how long each shared border is, so a model can treat a long
shared boundary as a stronger link than a short one.

**Link.** The connectivity is summarised into a few simple numbers (total states,
number of bordering pairs, average border length) that travel into the main feature
table.

---

## Pattern 2 — States Vary Hugely in Size and Shape

**Main idea.** Malaysian states range from tiny to enormous, and from compact to
elongated.

**Evidence.** Sarawak is by far the largest (~124,000 km²), followed by Sabah
(~73,000 km²). Peninsular states run from about 1,000 km² (Perlis) to ~36,000 km²
(Pahang). Some states are stretched and thin (Kelantan, Kedah), others compact
(Melaka, Perlis).

**Analysis.** Checking these sizes and shapes is also a data-quality step: it confirms
all 16 boundaries were read in correctly with no geometry errors. The cleaning script
explicitly verifies that exactly 16 units are present.

**Link.** Each state's centre point (centroid) is also computed and stored, ready for
distance-based map work in future model versions.

---

## What Goes Into the Model

Three simple numbers summarise the geography for the main feature table: the state
count (16), the number of bordering state pairs, and the average shared-border length.
These give the models a compact sense of the country's spatial connectivity
(Song et al., 2024; Wang et al., 2024).

---

## References

Song, J., Ding, J., Gui, X., & Zhu, Y. (2024). Assessment and solutions for vulnerability of urban rail transit network based on complex network theory: A case study of Chongqing. *Heliyon, 10*(5), e27237. https://doi.org/10.1016/j.heliyon.2024.e27237

Wang, Z., Huang, K., Massobrio, R., Bombelli, A., & Cats, O. (2024). Quantification and comparison of hierarchy in public transport networks. *Physica A: Statistical Mechanics and Its Applications, 634*, 129479. https://doi.org/10.1016/j.physa.2023.129479
