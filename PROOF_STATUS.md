# Collatz Proof Status Audit

Date: 2026-04-28

## Bottom Line

The repository is now cleanly classified as a conditional Collatz proof program, not a completed unconditional proof. I could not turn the current arguments into a fully rigorous proof of the Collatz conjecture, because every route from the existing evidence to all positive integers still uses an unproven pointwise transfer principle.

The central issue is not a missing computation. It is the gap between statements that hold for almost all integers, density-one sets, measure-one sets, finite verified ranges, or modular Markov chains, and the universal statement required by Collatz.

## Verified External Context

- David Barina, "Improved verification limit for the convergence of the Collatz conjecture," The Journal of Supercomputing 81, article 810 (2025): verifies convergence for all starting values below `2^71`.
- Terence Tao, "Almost all orbits of the Collatz map attain almost bounded values," Forum of Mathematics, Pi 10:e12 (2022): proves an almost-all result in logarithmic density, not a universal theorem.
- Krasikov and Lagarias, "Bounds for the 3x+1 Problem using Difference Inequalities," Acta Arithmetica 109 (2003): proves at least `x^0.84` integers below `x` eventually reach 1.
- Christian Hercher, "There are no Collatz-m-Cycles with m <= 91," Journal of Integer Sequences 26 (2023): eliminates nontrivial `m`-cycles up to `m = 91`, but not all possible cycles.

## Strongest Rigorous Reduction In The Repo

The clean theorem is:

```text
If Lemma UB holds, then the Collatz conjecture holds.
```

Lemma UB states that there are constants `C, alpha > 0` such that every `n >= 2` has first descent time

```text
delta(n) <= C * (log_2 n)^alpha.
```

The implication is rigorous. If every `n` eventually drops below itself, then strong induction proves every orbit reaches 1, using the verified finite base range as a computational base case.

## What Is Proven Or Computed

- Finite verification: all `n < 2^71` are externally verified by Barina 2025.
- Local computational evidence: the workspace reports first-descent and survival-decay checks through finite scales such as `B <= 32` or `B <= 29`, depending on the script.
- Almost-all theory: Tao-style and 2-adic/Borel-Cantelli style arguments support density-one or measure-one descent, not descent for every integer.
- Cycle exclusions: known cycle bounds eliminate many cycle types and ranges, but they do not eliminate all possible enormous cycles and do not address divergent orbits.
- Conditional algebraic reformulations: Numen and C*-algebra statements are useful reformulations or conditional routes, but the repo does not prove their required global hypotheses.

## Why The Current Proof Is Still Conditional

1. Density zero is not emptiness. A density-zero exceptional set can still contain infinitely many positive integers.
2. Measure-one 2-adic behavior is not a pointwise theorem for each embedded integer.
3. Spectral gap or Markov-chain mixing controls distributions, not the deterministic orbit of a fixed integer unless an additional transfer lemma is proved.
4. GPU verification up to any finite `B` cannot imply the statement for all `B` without a monotone certificate or induction principle.
5. Cycle elimination does not rule out nonperiodic divergent orbits.
6. Lemma UB, Conditional Survival Decay, the Transfer Hypothesis, Numen absolute continuity, and C*-irreducibility are not proved here for all required scales.

## Exact Missing Theorem

Any complete proof from this program needs a theorem of the following kind:

```text
For every integer n > 1, the deterministic Collatz orbit of n must enter
a verified descent channel in finite time, with a bound strong enough to
iterate by induction.
```

The current best version is Lemma UB. The current most concrete experimental route is Conditional Survival Decay, but it still needs a proof for all bit lengths and all hard residues.

## Most Promising Next Attack

The most actionable path is to convert Conditional Survival Decay from empirical evidence into a theorem. A viable proof would need:

1. A scale-uniform description of how adding a high bit changes the next `cB` valuations.
2. A deterministic or adversarial mixing bound, not only an average-case Markov bound.
3. A counting argument showing survival classes shrink below one possible integer after enough scale lifts.
4. A finite base case aligned with the proven induction, ideally using the external `2^71` verification.

## New Rigorous Theorem File

The file `REPELLER_EJECTION_THEOREMS.md` records new unconditional theorems proved during the proof search. The main result is an exact ejection theorem for repeated high-growth parity corridors: every such corridor is locked to a negative 2-adic repeller, and the ejection time is exactly the 2-adic distance from that repeller divided by the block length.

This proves that no positive integer can have an eventually periodic high-growth parity tail. It also proves that no uniform ejection bound depending only on a high-growth block can exist, because arbitrarily long finite corridors are forced by a single residue class modulo a high power of two.

## GPU Certificate Search

The file `GPU_CERTIFICATE_RESULTS.md` records a CUDA-backed finite-depth certificate run using `collatz_certificate_search_gpu.py`. With the verified base `2^71`, the offset domination theorem certifies every low-growth parity cylinder through depth 44. The depth-44 GPU audit scanned `2^24` split prefixes and matched exact CPU binomial arithmetic, certifying 16,746,514,229,214 of 17,592,186,044,416 depth-44 cylinders.

The remaining 845,671,815,202 cylinders are exactly the high-growth words with `3^k >= 2^44`. This confirms that the finite-depth low-growth route is closed at this scale; the hard obstruction is the aperiodic high-growth tail.

The file `PREFIX_BARRIER_THEOREMS.md` strengthens this obstruction. Any non-descending orbit above `2^71` must have every prefix through depth 183 remain high-growth, using the exact late-ones affine threshold. The CUDA audit in `collatz_prefix_barrier_gpu.py` reduces the depth-62 obstruction to 7,392,913,791,491,669 prefix-barrier words, about 0.1603082639 percent of all depth-62 words; the exact big-integer dynamic program gives a depth-183 barrier fraction below `2^-17.2468904772`.

The file `MECHANICAL_BARRIER_THEORY.md` identifies the structure of this remaining language. The survivor prefixes are exactly the binary words whose prefix sums dominate the lower mechanical word of slope `log_3(2)`. For a verified base `2^B`, exact threshold arithmetic gives a safe depth `D_B`; in particular `D_71 = 183`.

The file `INTERVAL_GPU_RESULTS.md` records complete CUDA scans of bit intervals through `[2^30, 2^31)`. These scans show that some integers remain inside the mechanical barrier through the safe depth, but all tested intervals still descend by step 700. The hardest scanned interval `[2^30, 2^31)` has maximum first descent time 433 at `n = 1,827,397,567`.

The file `DESCENT_TRIGGER_THEOREM.md` gives an exact first-descent criterion: `T^l(n) < n` if and only if the parity prefix is low-growth and its affine threshold `b_l / (2^l - 3^k_l)` is below `n`. This is now the sharp form of the missing theorem.

The file `BARRIER_DESCENT_EQUIVALENCE.md` records the strongest computational pattern: over every complete CUDA-scanned interval through `[2^30, 2^31)`, first descent equals first mechanical-barrier failure with zero mismatches. It reduces the remaining work to two precise claims: every positive integer eventually fails the mechanical barrier, and every finite first-failure word has a descending least positive residue, except the known `1` cycle equality case. The exact verifier `collatz_first_failure_verify.py` checks that finite-word lemma through length 39, covering 408,434,812 first-failure words with no bad cases.

The file `FIRST_FAILURE_RESIDUE_THEOREM.md` proves a new exact reformulation of the finite word gap. Every nontrivial first-failure word is a critical high-growth survivor followed by `0`, and its least-residue descent is equivalent to `(2^l - 3^k)y >= b`, where `y` is the dual output residue defined by `2^l y = b mod 3^k`. This removes the least positive cylinder representative from the proof target and replaces it with a modular lower-bound problem.

The file `STURMIAN_WEDGE_PROGRAM.md` sharpens that finite target again. In each first-failure jump length, the multiplier and modular inverse are fixed, so the residue lemma is equivalent to proving that the structured affine-offset set cut out by the mechanical dominance language avoids the open modular wedge `d y < b`. A near-critical counterexample outside the dominance language shows the prefix condition is essential. The script `collatz_wedge_exchange.py` stress-tests local exchanges inside this dominance language and shows that adjacent valid `01 -> 10` moves are not monotone in the wedge margin, so the finite proof needs a floor-invariant or block-exchange argument rather than a simple majorization proof. The script `collatz_offset_membership.py` attacks the exact forbidden wedge from the output side using 3-adic offset decoding. It also records an important correction: the stronger shortcut `Y_l > floor(b_mech/d)` fails at some later layers, but the exact open-wedge condition can still remain empty. With the refactored decoder, every jump layer from `81` through `120` has now been checked from the forbidden side: `12,003` exact forbidden offsets, zero hits.

The file `APERIODIC_TAIL_OBSTRUCTION.md` proves a new inverse-coding dichotomy for the infinite obstruction. In Bernstein-Lagarias parity coordinates, any positive non-descending orbit in the mechanical dominance cone must either have a divergent inverse parity series or have a convergent inverse series with a non-vanishing normalized escape term. This explains why the remaining obstruction is genuinely aperiodic and cannot be removed by the fixed-repeller ejection theorem alone.

This infinite obstruction also matches the known critical-density boundary from López and Stoll's 2-adic conjugacy work: supercritical aperiodic parity vectors are mapped to aperiodic 2-adic integers, while any rational non-cyclic trajectory must have lower `1`-density exactly `log_3 2`. The mechanical barrier is precisely this boundary case, so the remaining problem cannot be solved by a supercritical-density argument alone.

The file `CONTRADICTION_THEOREM_PROGRAM.md` records the current non-computational attack. It proves that a finite open-wedge obstruction is exactly a constrained 3-adic peel chain with forced parity locks and monotone residual windows. It also proves an exact normalized-escape increment identity: the inverse parity series is precisely the monotone pressure added to `R_L = 2^L T^L(n)/3^sigma`. The remaining bridge theorems are now stated as a peel-chain no-crossing theorem and a critical escape exclusion theorem.

The file `TWO_ADIC_STABILIZATION_OBSTRUCTION.md` sharpens the infinite bridge theorem into a purely pointwise 2-adic statement. For any infinite parity word, its compatible cylinder residues have binary carry bits `eta_L`; the word is the parity sequence of a nonnegative integer if and only if these carries are eventually zero. Therefore an infinite mechanical-tail counterexample is equivalent to an eventually stabilizing 2-adic lift inside the mechanical dominance cone. A proof of mechanical carry forcing, namely that every infinite mechanical-cone word has infinitely many nonzero lift carries, would close the infinite aperiodic gap.

The file `PROOF_ROUTE_AUDIT_AND_RESEARCH_PLAN.md` is now the route-control note. It lists each existing proof path, what it actually proves, why it remains conditional or incomplete, which shortcuts have already failed, and which two bridge theorems remain live: peel-chain no-crossing and mechanical carry forcing.

The file `NEW_THEORY_RESEARCH_PROGRAM.md` records the current research plan for inventing a genuinely new unconditional theory. Its main candidate is bi-adic Sturmian obstruction theory: prove that standard Sturmian blocks at continued-fraction scales of `log_3 2` force nonzero 2-adic parity-lift carries at unbounded scales. If its nested congruence obstruction can be proved, it would close the infinite mechanical-tail bridge without relying on probability or finite computation.

The file `STURMIAN_CARRY_PROOF_ATTEMPT.md` starts that proof attempt. It proves the local finite-block carry-congruence lemma and identifies why the original standard-block lemma is too weak alone: every finite prefix has a compatible parity-cylinder residue. The corrected infinite target is residue growth of `rho_L` along continued-fraction return scales.

The file `RETURN_BLOCK_RESIDUE_GROWTH_OBSTRUCTION.md` records the direct attempt to prove that corrected target. It proves that return-block residue growth is equivalent to mechanical carry forcing and to the absence of positive integer parity sequences inside the mechanical dominance cone. It also records the failed direct proof routes and the exact missing ingredient: an invariant that forces least nonnegative cylinder residues to grow despite balanced exchanges and accumulated excess.

The file `INFINITY_TO_FINITE_PROOF_METHODS.md` researches proof methods that really handle infinity in other areas and asks whether they can be replicated here. Its conclusion is that the strongest finite-scale analogue is renormalization on Ostrowski return blocks, while the strongest direct infinite route is a Subspace-Theorem-style collapse of infinitely many good scales to finitely many algebraic templates.

The file `EXTERNAL_COLLATZ_RESEARCH_AUDIT.md` adds a source-backed external audit. It distinguishes accepted partial results from unverified full-proof claims, and it confirms that the most relevant accepted structural work is the 2-adic conjugacy / critical-density line of Bernstein-Lagarias, Rozier, and López-Stoll. It also confirms that Tao's strongest accepted result is still an almost-all theorem, not a pointwise proof.

The file `FULL_PROOF_ROUTE_ATTEMPTS.md` now tries all five ranked routes in exact theorem form. For each route it states a precise theorem schema and proves that, if that route-specific theorem is achieved, then the corresponding bridge theorem follows. The synthesis is that the renormalized Ostrowski cocycle route is the only one with a plausible path to both bridges in one language; the subspace and rotation routes are best seen as infinite-bridge routes, and the well-quasi-order route is best seen as a finite-bridge route.

The file `OSTROWSKI_RETURN_BLOCK_COCYCLE.md` now defines the exact Route 1 return-block state space and proves the first induced block-map lemma. It shows that the remaining Route 1 problem is no longer how to define the cocycle, but how to compress the exact block summaries and boundary states to a finite quotient with a coercive Lyapunov drift.

That note now also tests the first fixed-truncation active quotient. A fixed truncation `Q_r(u) = (e(u), sigma(u) mod 2^(r-2), y(u) mod 2^r)` is shown not to be closed under balanced exchanges: two length-19 floor-critical balanced-exchange prefixes have the same `Q_5`, but the same appended length-8 floor-critical block sends them to different next `Q_5` states. This narrows the remaining Route 1 problem again: any successful quotient must keep scale-growing phase information or an equivalent scalar summary of the high-bit quotient term.

The same Route 1 note now proves the exact adaptive one-block closure theorem. If the next block has length `m`, then writing `y = eta_m + 2^m lambda_m` gives an exact update formula `q = lambda_m + kappa_m`, where `kappa_m` is determined by the low `m` bits and the block data. So the precise remaining Route 1 bottleneck is now: control the adaptive scalar `lambda_m` by a Lyapunov-type inequality, or replace it by a bounded surrogate whenever the ordinary residues stabilize.

That bottleneck has now sharpened further. The note proves that the correction term `kappa_m` is always nonnegative, so `q >= lambda_m` in general. It also proves that for nondeficit blocks `k(v) >= h(m)`, the same-depth quotient `floor(y' / 2^m)` is at least `lambda_m`, giving the first exact local Lyapunov-type inequality for `lambda_m`. Finally, under ordinary residue stabilization and actual continuation blocks, the correction term collapses completely: `z_m = 0`, `kappa_m = 0`, and `q = lambda_m`. The remaining open step is now very specific: turn these local facts into a return-scale theorem forcing enough nondeficit blocks, or derive a contradiction between long-term stabilization and the repeated same-depth growth of `lambda_m`.

That same Route 1 note now adds the first exact refinements in both directions. In the stabilized zero-correction regime, the same-depth update becomes

```text
lambda_m^+ = lambda_m + floor((y(v) + (3^{k(v)} - 2^m) lambda_m) / 2^m),
```

so equality on a nondeficit block can persist only below an explicit threshold. Separately, the convergent arithmetic of `h(q_s) = ceil(q_s log_3 2)` is now exact: for convergents `p_s / q_s`, one has `h(q_s) = p_s + 1` on even scales and `h(q_s) = p_s` on odd scales, hence the total child deficit in the standard `a_s q_{s-1} + q_{s-2}` partition of a floor-critical prefix is exactly `0` for even `s` and exactly `-a_s` for odd `s`. This gives the first rigorous continued-fraction nondeficit statement: every even convergent scale has at least one nondeficit child block.

The Route 1 note now sharpens that again. If `D_j` is the cumulative deficit of the first `j` long child blocks in the standard continued-fraction partition of a floor-critical prefix of length `q_s`, then `D_j >= h(j q_{s-1}) - j h(q_{s-1})`. Explicitly, `D_j >= 0` on even parent scales and `D_j >= 1 - j` on odd parent scales. This gives the first ordered return-scale control, not just an existence count. It also yields a genuine strict-drift corollary: on every odd return scale, the first long child block has length `q_{s-1}` with even convergent index, and in the stabilized regime it forces strict same-depth growth `lambda_{q_{s-1}}^+ >= lambda_{q_{s-1}} + 1` whenever `lambda_{q_{s-1}} > 0`.

At the same time, the note proves an exact limitation of the current method. The coarse equality thresholds `B(q_r) = floor((2^{q_r} - 1) / (3^{h(q_r)} - 2^{q_r}))` are zero on the even convergent lengths that feed those odd-return strict-drift steps, but they grow without bound on the odd convergent subsequence. So the local `lambda` threshold mechanism alone still cannot prove that equality occurs only finitely often across all scales. The remaining Route 1 gap is now asymmetrical and explicit: propagate positivity of `lambda` onto infinitely many odd return scales, or add a second ingredient that turns ordered long-prefix nondeficit control into scale-to-scale positivity.

The note now also identifies that second ingredient in exact form. It defines an exceptional zero graph `G_m` on states `x < 2^m` whose depth-`m` parity block is nondeficit and whose depth-`m` output still lies below `2^m`. Under stabilized actual continuation, the zero regime `lambda_m = 0` is exactly a path in `G_m`: the current orbit value is the least residue of the next block, and zero-to-zero continuation is the edge `x -> T^m(x)`. This turns the vague zero-regime problem into a concrete finite-state obstruction.

That reformulation immediately yields a new limitation result. At the first odd return scale `q_5 = 19` with child depth `8`, ordered long-prefix deficit control does not force positivity even under the exact stabilized compatibility condition. There are exactly three floor-critical length-19 words whose first two long children form compatible zero-regime paths in `G_8`, with state paths `63 -> 182 -> 175` and `27 -> 242 -> 233`. So the remaining Route 1 theorem is no longer “prove positivity from deficit counts”; it is the sharper exceptional-tube exclusion statement: prove that stabilized mechanical-cone continuations can visit the exceptional zero graphs `G_{q_{s-1}}` on only finitely many odd return scales.

That obstruction also starts later than it first appeared. Because any actual counterexample must lie above the externally verified base `2^71`, every non-descending orbit value on that counterexample is also above `2^71`. Hence `lambda_m > 0` is automatic for every `m <= 71`, so the zero graphs `G_8` and `G_65` are abstract Route 1 obstructions but not live obstructions for an actual counterexample. The first odd-return child depth whose zero regime is not excluded by size alone is `q_8 = 485`.

The new exact analyzer `collatz_exceptional_zero_graph.py` exposed a stronger structural fact: every edge of every exceptional zero graph `G_m` is strictly increasing, so each fixed `G_m` is acyclic. Extending that exact computation with the symbolic prefix-barrier successor builder now sharpens and slightly corrects the earlier lifetime picture. On the prefix-barrier slice, the observed maximum transient length stays at `3` through `m = 32`, but the first depth with a four-block transient is `m = 33`. That longer class is still extremely sparse in the currently tested range: at `m = 33` the path-length histogram is `20677836, 50140, 125, 1` for lengths `1, 2, 3, 4`, and at `m = 34` it is `33084529, 64159, 109, 1`. So the current evidence still strongly disfavors long same-depth zero-tube persistence inside a fixed graph, but the old provisional slogan "dies within at most three depth-`m` blocks" is no longer correct beyond the previously enumerated range. The remaining Route 1 obstruction is still cross-scale re-entry into the first-child exceptional zero regime at large odd return depths such as `485`, not recurrence inside a fixed zero graph.

That obstruction is now decomposed one step further. The new script `collatz_first_child_zero_dp.py` proves the exact q5 forced-prefix formula computationally: once a first long child enters the zero regime, any further same-depth zero blocks are forced by the actual orbit of the zero state, and only the remaining tail is combinatorial. At `q_5 = 19 = 2 * 8 + 3`, forcing two zero blocks leaves a 3-bit suffix and reproduces the exact count `3`. At the first live case `q_9 = 1054 = 2 * 485 + 84`, a two-block zero tube would force the first `970` bits and leave only an `84`-bit symbolic suffix. So the live Route 1 task is now an exact finite-state plus suffix-count problem at depth `485`, not an undifferentiated infinite zero-regime problem.

That q9 tail is not generic. The Route 1 note now proves the exact shifted-floor identity `h(970 + t) = 613 + h(t) - 1` for `1 <= t <= 84`. Therefore a two-block depth-`485` zero tube with minimal forced odd count `613` leaves an `84`-bit shifted first-failure layer, directly connecting the first live odd-return obstruction to the first-failure verification machinery.

That reduction now extends across the whole second depth-`485` block. The note proves `h(485 + t) = 307 + h(t) - 1` for every `1 <= t <= 485`. So if a two-block zero tube has minimal forced odd count `K_2 = 613` and the first block contributes `K_1 = 307 + e_1`, then the entire second block is a shifted `(e_1 + 1)`-failure layer of length `485`. The new script `collatz_recursive_partition_dp.py` then builds the exact pre-bit search surface on the internal partition `485 = 5 * 84 + 65`: it computes boundary-shift transition counts blockwise, validates on the small partition `8 = 3 + 3 + 2`, and resolves the minimal second-block counts by first-child excess `delta`. In the currently validated output, the `delta = 0` branch is already nonempty for `e_1 = 0, 1, 2`, so the splice-theorem branch is a real surviving obstruction, not a vacuous corner case.

The next local theorem is now split exactly rather than overclaimed. The note proves the child-splice threshold formula: for a contracting first child `u_1`, a realization `rho(u_1) + 2^84 lambda` descends exactly when `lambda >= lambda_crit(u_1)`. But the blanket local splice claim is false once the first child has positive excess `delta > 0`: the exact length-12 word `101010101010` gives a shifted-failure parent with a contracting first child `10101`, positive excess `delta = 1`, threshold `lambda_crit = 1`, and actual splice quotient `lambda_actual = 0`. So the live 84-child problem no longer has a single theorem target. Instead it splits into two exact branches. If `delta = 0`, then the 401-bit suffix is floor-critical with fixed odd count `h(401) = 254`, and the splice quotient is the exact residue `lambda_84(v_2) = a_1^(-1)(rho(z) - y(u_1)) mod 2^401`. Because each residue class modulo `2^L` determines a unique parity word of length `L`, the new script `collatz_floor_critical_residue_verify.py` can verify the forbidden `delta = 0` classes directly by reconstructing the unique suffix word attached to each residue; its brute-force small self-tests pass exactly. Because the suffix multiplier is constant on that layer, this is equivalently a finite forbidden-class problem for the suffix residue `rho(z)` or suffix offset `b_z mod 2^401`, not a first-failure lemma at length `401`. If `delta > 0`, then the suffix is a shifted `delta`-failure layer of length 401, so the obstruction recurses to a shorter shifted-failure block. The script `collatz_shifted_failure_splice.py` now computes the suffix-driven splice quotient directly and validates it against the full-word residue formula. The corrected small-model evidence remains: no counterexamples were found in the tested `delta = 0` families, but positive-`delta` counterexamples do occur. The new exact q5 reachability overlay makes the missing state constraint concrete: for the two first-block groups arising from the three actual q5 parent words, the generic second-block surfaces have sizes `121` and `70`, while the actually reachable second-block sets collapse to a single word in each group. After lifting that overlay to the internal partition `[3, 3, 2]`, the actual second-block start states are exactly `242` and `182`, and the actual boundary-shift histograms collapse to singleton tracks inside the much larger generic boundary surfaces.

That q5 collapse still points toward a real state constraint, but one earlier formulation overstated what the finite evidence shows. On the minimal two-block layer, once the first block is fixed, the actual second block is just the deterministic next depth-`m` orbit segment of that state. So singleton actual boundary-shift support for a fixed actual second-block start state is largely automatic on that slice and is not, by itself, the missing transfer theorem. What the exact computations really show is sparser but still meaningful: on the minimal two-block prefix-barrier layer, the actual two-block set remains a small subset of the full zero layer, for example `1712 / 57155` at `m = 24`, `27291 / 3438211` at `m = 30`, `130388 / 7773196` at `m = 32`, and `212765 / 33148798` at `m = 34`.

So the sharp open problem is now stated more honestly. The nontrivial missing theorem is not uniqueness of the boundary-shift track once an actual second-block start state is known; that is built into the deterministic dynamics. The missing theorem is a compressed structural description of which second-block start states are actually reachable on the canonical Ostrowski partition at the first live scale `485 = 5 * 84 + 65`, or an equivalent theorem that collapses the generic q485 partition surface to finitely many actual 84-bit first-child prefixes that can then be fed into the 401-bit forbidden-residue verifier. That remains the sharp Route 1 gap.

The new actual-only compressor `collatz_actual_start_state_compressor.py` now gives the first exact return-scale evidence for that sharper formulation. At `q_5 = 19`, `q_6 = 65 = 19 + 19 + 19 + 8`, and `q_7 = 84 = 19 + 19 + 19 + 19 + 8`, the tested actual layers have `2`, `51`, and `8` reachable second-block start states respectively, with `0`, `3`, and `1` multi-profile start states. At `q_6` the exact actual layer has `2444` parent words but only `51` reachable second-block start states and only `14` distinct boundary profiles. At `q_7` the same compressor collapses further to `182` actual parent words, `8` reachable second-block start states, and `8` distinct boundary profiles on `[19, 19, 19, 8]`. The only remaining ambiguity at these tested return scales is small and structured: some start states support two profiles, but in every tested case the pair `(second-block start state, first-block odd count)` determines a unique boundary profile. Concretely, at `q_6` the only multi-profile states are `19273`, `123634`, and `465922`, and at `q_7` the only multi-profile state is `19273`; in each case the two profiles are separated exactly by first-block odd counts differing by `1`, and the profile itself shifts by `(+1, +1, 0)` at `q_6` or `(+1, +1, +1, 0)` at `q_7`.

The generic q485 partition surface is still much larger, but the new boundary-support DP has now measured its exact width on the canonical partition `485 = 84 + 84 + 84 + 84 + 84 + 65` without enumerating generic second-block words. For first-block excess `e_1 = 0, 1, 2, 3`, the support ranges at the successive boundaries are exactly `0..(32 + e_1)`, `0..(63 + e_1)`, `0..(94 + e_1)`, `0..95`, `0..42`, and finally `0` at the full length `485`. Equivalently, the support sizes are `33 + e_1`, `64 + e_1`, `95 + e_1`, `96`, `43`, and `1`. So the last `65`-block already forces a strong uniform collapse on the generic partition surface before the final exact closure, which is the first clear mathematical sign that the q485 obstruction may be compressible at the partition level rather than only by brute-force overlay.

The new script `collatz_q485_prefix_candidates.py` now carries out the first exact finite handoff from the tested `84`-bit actual surface to the live q485 partition. It uses the exact q7 actual surface of `182` words, grouped into `9` buckets by the corrected variable `(second 19-block start state, first 19-block odd count)`, and lifts those buckets into the q485 branches `e_1 = 0, 1, 2, 3`. On that finite actual surface, every branch survives the exact `420`-boundary cap `0..42`: for each `e_1`, the compatible q485 branch is `delta = e_1 + 1`, the boundary-420 support size is exactly `43`, and all `182` actual `84`-bit prefixes survive. But every one of those prefixes is automatically noncontracting, because each has odd count `53 = h(84)`, so its denominator `2^84 - 3^53` is nonpositive. Thus the current finite q485 handoff produces `0` contracting prefixes and `182` noncontracting prefixes. In other words, this exact handoff reaches only the positive-`delta` recursive branch, not the direct `delta = 0` splice-theorem branch.

That finite handoff now admits a sharper exact compression than the raw `182`-word list. On the tested q7 actual surface, those `182` words collapse to `9` deterministic trunks. Each corrected bucket `(second 19-block start state, first 19-block odd count, boundary profile)` comes from a single actual zero-state start, a single four-step actual `19`-block state path, and a single forced `76`-bit prefix, leaving only an `8`-bit suffix symbolic. So on the currently exact `84`-bit surface the right state object is already a trunk

`actual start state + actual 19-block state path + forced 76-bit prefix + 8-bit suffix layer`,

not an arbitrary `84`-bit word.

The positive-`delta` recursive branch is now reduced and closed exactly for the currently live q485 starts. Starting from the live suffix length `401` with start shifts `1, 2, 3, 4`, the first recursive `84`-child allows delta ranges `0..32`, `0..33`, `0..34`, and `0..35` respectively, so naive recursion still expands to millions of admissible delta paths. At the `84`-scale alone, all four live q485 branches collapse to the same exact set of `46` terminals: four shorter `delta = 0` splice interfaces at remaining lengths `317`, `233`, `149`, and `65`, together with the `42` positive-shift base terminals at length `65` with shifts `1..42`. Refining that remaining `65`-bit base on its canonical partition `65 = 19 + 19 + 19 + 8` collapses the same four live starts even further to a uniform terminal set of size `13`: direct `delta = 0` splice interfaces at lengths `317`, `233`, `149`, `65`, `46`, `27`, and `8`, together with the six base `8`-bit positive shifts `1..6`.

Those `13` terminals are all harmless. The seven `delta = 0` terminals are vacuous because the corresponding floor-level first-child denominators are already negative at the relevant split scales: `2^84 - 3^53 < 0`, `2^19 - 3^12 < 0`, and `2^3 - 3^2 < 0`, so no such floor-critical first child is contracting. The remaining six positive base terminals are exactly the `8`-bit shifted-failure layers with shifts `1..6`, and exhaustive classification under the canonical `3 + 3 + 2` split finds zero splice failures in all six cases. Therefore the positive-`delta` branch closes completely on the current q485 handoff. In particular, the exact finite q7 lift of `182` noncontracting `84`-bit prefixes no longer represents a live unresolved obstruction: every one of those prefixes feeds only into the now-closed positive-`delta` side.

This also clarifies the current computational blocker. A direct q485 actual compressor built from depth-`84` prefix-barrier parity-word enumeration is not viable with the present symbolic generator: the exact prefix-barrier language at depth `84` already has size `10,827,865,726,573,282,854,352`. So the remaining Route 1 work at q485 is not just to run more code. It is to find a new exact state compression for the live `84`-bit child surface that bypasses parity-word enumeration and distinguishes the `delta = 0` splice branch from the positive-`delta` recursive branch.

The file `PEEL_STATE_WQO_PROGRAM.md` now starts the finite backup route in exact form. It defines a candidate block-level quasi-order on enriched peel states, proves that the order is a well-quasi-order, and proves the finite-basis reduction that would follow from upward-closure of bad states. It also isolates the exact missing monotonicity statement as an open block-lift theorem.

Until that transfer theorem is proved, the honest status is: strong conditional proof architecture, strong computation, no complete rigorous proof.
