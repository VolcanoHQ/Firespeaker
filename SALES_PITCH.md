# Firespeaker — Investor Pitch

*Volcano Studios (GitHub: [VolcanoHQ](https://github.com/VolcanoHQ)) · July 2026*

---

## The one-liner

> Producing one audiobook costs $5,000–$15,000 and takes months. Firespeaker turns a
> manuscript into a fully-produced audiobook — narration, cast voices, music, sound
> design — in about an hour, at near-zero marginal cost.

**The bigger arc:** the same manuscript-understanding engine climbs from audio
into video. Audiobooks (Tiers 1–3) are the shipping product; illustrated,
animated, and full screen adaptations (Tiers 4–6) are the roadmap — one
pipeline, six formats, from the same book.

---

## 1 · The problem

- 4M+ books are published every year; fewer than 5% ever become audiobooks.
- Production cost is the gate: studio time, narrator fees, engineering, mastering.
- Indie authors and publisher backlists are locked out entirely — an enormous
  catalog of finished books with zero audio presence.

## 2 · The product

**EPUB in → one button → chaptered audiobook out.**

Three production tiers shipping today, chosen by the author:

| Tier | What you hear | Analogy |
|---|---|---|
| **Tier 1** | Single narrator, ACX-compliant master | Standard audiobook |
| **Tier 2** | Narrator + a distinct voice for every character | Full-cast recording |
| **Tier 3** | Cast + scored music, ambience, layered sound effects | Graphic-Audio-style audio drama |

Plus a browser studio: review and correct the analysis (speaker attribution,
scene boundaries), preview any tier on a single scene before committing, and —
for Tier 3 — a multi-lane **scene mixer** with per-asset regeneration, giving
authors director-level control instead of a black box.

*The demo is the pitch: play 60 seconds of a Tier 3 master, then click Generate.*

## 3 · The customer case: cheaper, faster, more selection

Amazon was built on the three things customers will still want in ten years:
**lower prices, faster delivery, bigger selection.** Audiobook customers are
no different — and traditional studios are structurally unable to deliver any
of the three, because their unit of production is scarce human hours in a
physical room. Ours is rented compute.

**Cheaper.** $19–199 a title against $5,000–15,000 — 1–3% of the studio
bill. And the gap only widens: our costs ride the falling GPU curve; theirs
are salaries and studio time, which only go up.

**Faster.** About an hour from upload to chaptered master, against 3–6 months
of scheduling, recording, and editing. Revisions are instant and free instead
of pickup sessions booked weeks out.

**More selection — in both directions.** Bookstores had shelf space; studios
have calendar space. That's why fewer than 5% of books ever get audio: studios
ration scarce capacity to safe bestsellers, exactly as physical retail
rationed shelves. We make production capacity effectively infinite, so the
entire long tail — every indie title, every backlist book — gets produced.
That's the catalog side. The product side: not one narrator but a marketplace
of licensed voices cast by description, not one format but three tiers from
clean narration to a scored, sound-designed audio drama, with every line,
music cue, and effect customizable in the browser. The everything store
logic, applied to audiobook production.

Head to head:

| | Traditional studio | Volcano Studios |
|---|---|---|
| **Price** | $5,000–$15,000 per title | $19–$199 per title — **1–3% of the studio bill** |
| **Turnaround** | 3–6 months of scheduling, recording, editing | **~1 hour** from upload to chaptered master |
| **Casting** | One narrator, whoever's available and affordable | A **marketplace of licensed voices** — audition by description, give every character a distinct voice |
| **Formats** | Straight narration, take it or leave it | **Three tiers**: clean narration → full cast → scored, sound-designed audio drama (video editions on the roadmap) |
| **Revisions** | Pickup sessions billed by the hour, weeks later | **Regenerate any line or asset instantly**, free, in the browser |
| **Creative control** | Notes over email; the engineer decides | **You direct**: fix attribution, remix scenes on a timeline, swap music and effects |
| **Risk** | Pay up front, hear it when it's done | **Preview any tier on a real scene** before spending a dollar |
| **Access** | Gated by budgets and studio calendars | **Self-serve** — if you finished the book, you can afford the audiobook |
| **Ethics** | Rights often informal | Every voice **consent-verified with provenance**; actors are paid per use |

And the flywheel spins the same way Amazon's did:

> Lower prices bring more authors → more titles means more voice licensing
> revenue → more actors join the marketplace → better casting selection makes
> better productions → which bring more authors. Meanwhile volume amortizes
> our fixed costs and GPU prices keep falling — so prices drop again, and the
> wheel turns faster. A studio's flywheel runs in reverse: every cost input
> is human time, and it only gets more expensive.

The pitch to an author in one breath: *a traditional studio sells you one
narrator's months; we sell you a whole production company for the price of
dinner — and you're the director.*

For a publisher with a backlist, the same table reads as a spreadsheet: a
500-title catalog that was never economical to convert becomes a
five-figure project instead of a seven-figure one.

## 4 · The tier ladder IS the business model

Costs scale with the customer's willingness to pay, because AI spend scales the
same way:

- **Tier 1** is nearly free to produce (zero LLM calls on clean manuscripts;
  deterministic pipeline + local synthesis) → freemium / volume.
- **Tier 2** uses bounded, windowed LLM calls → subscription.
- **Tier 3** adds music/SFX generation → premium per-book pricing.
- **Paid plans map directly to premium model lanes** — already built: a
  project's plan selects its provider chain (free chain vs. paid fast lane),
  metered per user and per project.

## 5 · Unit economics & pricing

Compute is rentable per-second, so COGS is measured, not hand-waved. An
80k-word novel is ~9–10 hours of finished audio; at our measured RTF 0.484
that's ~4.5 GPU-hours, and a rented RTX 4090 runs ~$0.40/hr on serverless
marketplaces (RunPod, Vast, Modal). Per-title cost of goods:

| Tier | COGS per book | Suggested price | Gross margin | % of traditional cost |
|---|---|---|---|---|
| **Tier 1** | ~$1.50–2.50 (GPU only; zero LLM calls on clean manuscripts) | $19–29, or free with limits | ~90% | **<1%** of the ACX floor (~$2,000–4,000 at $200–400/finished-hour) |
| **Tier 2** | ~$2–4 (adds bounded, windowed LLM calls) | $49–79, or bundled in subscription | ~93% | **~2%** of a multi-voice recording |
| **Tier 3** | ~$4–8 (adds music, ambience, SFX generation + mastering) | $99–199 per book | ~95% | **1–2%** of a $5K–15K Graphic-Audio-style budget |

Pricing logic: charge **~1–3% of the equivalent traditional production** —
cheap enough to be an impulse purchase for an indie author, while every tier
still clears 90% gross. As traditional prices anchor the market, we price as a
percentage of them, not as cost-plus.

Why the cost side stays low and keeps *falling*:

- **Serverless per-render GPUs.** Renders dispatch to per-second-billed cloud
  GPUs — the pipeline already runs as detached, restartable jobs, and the paid
  fast lane is just a plan→compute-lane mapping on entitlement machinery we've
  already built. Zero idle hardware.
- **Spot instances + per-chapter checkpointing** cut GPU spend roughly in half
  again; the line-level synthesis cache makes resume-after-preemption nearly
  free.
- **Tier 1 analysis costs zero AI dollars** on clean manuscripts —
  deterministic pipeline plus free-tier LLM quotas cover the gates.
- **GPU prices only go down.** Our COGS rides the hardware curve; traditional
  narration costs don't.

One dependency, stated plainly: paid renders wait on the commercial model swap
(see Honest risks) — a funded work item, not a research risk.

## 6 · Why now

- Open TTS and audio-generation models crossed the quality threshold in the
  last 18 months.
- We render **faster than realtime on a single laptop GPU** (measured synthesis
  RTF 0.484 on an RTX A3000 — a novel renders in roughly an hour, not overnight).
- Free-tier LLM quotas are sufficient for the entire analysis pipeline, so
  cost-of-goods at Tier 1 approaches zero.

## 7 · The moat

Not the models — everyone has those. Three things nobody else has together:

1. **Deterministic-first pipeline with measured accuracy.** Structure analysis
   runs with *zero* LLM calls on clean manuscripts; AI engages only as a gate
   on failure signals. Measured against human-cleaned gold references:
   - Text fidelity: **99.5–100%** coverage, ≤0.31% contamination
   - Dialogue speaker attribution: **99.2%** (124/125) and **100%** (5/5)
   - This is quality no raw-LLM pipeline delivers, and it's regression-tested.
2. **A consent-based performance marketplace.** It starts with voice: actors
   record one ~15-minute session, publish a licensed voice with SHA-256
   consent provenance, and get paid per use. The same consent-ledger +
   per-use-payout machinery generalizes up the tier ladder — likeness and
   on-camera performance for the illustrated/animated tiers, motion capture
   for the video tiers. This flips "AI steals creative work" into a revenue
   channel for the very performers AI threatens — a defensible supply-side
   network that deepens with every tier.
3. **The scene mixer.** Multi-lane timeline (voice / music / ambience / SFX),
   click-to-audition, per-event mute/gain/nudge, and per-asset regeneration
   ("new take" or author's own prompt). Authors direct; nobody else in the
   category offers this.

## 8 · Traction & evidence (pre-revenue)

- 8-book validation campaign across public-domain literature (Les Misérables:
  50 parts / 498 chapters / 723 scenes, fully deterministic).
- End-to-end proven: EPUB upload → analysis → correction console → one-button
  render → chaptered M4B with per-line timing manifest.
- Complete Tier 3 masters produced (e.g. a 45.6-minute production with 20
  music-state events and 68 generated sound-design assets).
- Full platform already built: auth, user-owned projects, per-project usage
  metering, plan-based provider entitlement, voice cloning studio, marketplace
  REST API.
- Next: guided alpha with 5–10 indie authors rendering their own manuscripts;
  headline metric — *"would you listen to a whole book made this way?"*

## 9 · The road to Tier 6: from audiobook to screen

The manuscript analysis engine — structure, scenes, cast, per-line timing,
scene-level direction (music, environment, dramatization) — is exactly the
pre-production a film needs. The audio tiers fund and prove it; the video
tiers cash it in:

| Tier | What you get | Status |
|---|---|---|
| **Tier 4** | Illustrated / motion-comic: Tier 3 audio + identity-locked scene stills, cut on the per-line timing track | Near-term — the assets already exist |
| **Tier 5** | Animated episode: image-to-video on those stills, lip-synced dialogue, camera direction | Roadmap — model-dependent |
| **Tier 6** | Screen adaptation: long-form video generation with cast and scene continuity — manuscript in, film out | Vision — where the category is headed |

Why this is credible and not vapor:

- **Character continuity is solved for stills today:** seed-locked character
  sheets produce identity-locked scene images (8/8 in validation) — the hard
  problem of "the hero looks the same in every scene."
- **The sync track already exists:** every rendered line carries a measured
  start/end timestamp. That timing manifest is a video edit decision list —
  visuals cut to it for free.
- **The screenplay already exists:** Tier 3's per-scene direction artifacts
  (cast, environment, dramatization, sound design) are structurally a shot
  list. Tiers 4–6 are renderers on top of it, not new understanding.

Every tier reuses everything below it. A competitor starting at video has to
rebuild our entire manuscript-understanding stack first; we get to ride the
video-generation model curve with the pre-production layer already built.

**The marketplace climbs the ladder too.** The voice marketplace's pattern —
guided capture session → measured QC → consent ledger → licensed listing →
per-use payout — extends to the performances video needs:

| Marketplace | Performer supplies | Powers |
|---|---|---|
| **Voice** (built) | ~15-min recording session | Character voices, Tiers 2–6 |
| **Likeness / acting** | Reference footage + expression session; casting by description, like voices today | Character appearance & screen presence, Tiers 4–6 |
| **Motion capture** | Movement library (gait, gestures, combat, dance) — phone-video mocap is already viable, no suit required | Character animation, Tiers 5–6 |

Actors don't get replaced by the advanced tiers — they get *cast* in them,
with the same consent provenance and payout bookkeeping the voice marketplace
runs today. A working actor can license voice, likeness, and movement as
three income streams on one profile.

## 10 · Business model summary

- **Authors/publishers:** freemium Tier 1 → subscription Tier 2 → premium
  per-book Tier 3; paid plans unlock faster premium-model lanes.
- **Voice actors:** marketplace listings; platform takes a transaction fee on
  voice licensing; per-use payout bookkeeping.
- **Performers (Tiers 4–6):** the same transaction-fee model on likeness and
  mocap licensing — three income streams per performer profile.
- **Long term:** backlist conversion deals with small/mid publishers (bulk
  Tier 1/2), Tier 3 as a differentiated consumer product, and Tiers 4–6 as
  the studio-licensing / adaptation-rights play.

## 11 · What the raise buys

1. Commercial licensing for the synthesis stack (swap/upgrade the
   non-commercial research models — path already scoped).
2. GPU render infrastructure + render queue (the pipeline already survives
   restarts; scaling it is engineering, not research).
3. Payments + marketplace hardening (payment processor, payout rails).
4. The first 100 authors: alpha → paid beta.
5. Tier 4 prototype (illustrated edition) — the first rung of the video ladder.

## 12 · Honest risks (say them before diligence finds them)

- **Model licensing:** current research models (XTTS, MusicGen) carry
  non-commercial terms — the swap to commercially-licensed equivalents is a
  funded work item and a launch gate for paid renders. No revenue flows
  through them until it's done.
- **Category noise:** "AI audiobook" is crowded at the TTS layer. We don't
  compete there — we sell the *production studio* (full-cast, scored,
  sound-designed) at text-to-speech prices, with a consent-based voice economy
  attached. Nobody else is selling Tier 3.
- **Rights:** audio output rights follow the author's text rights; the consent
  ledger covers the voice side.

---

## Appendix · Proof of work — how this was actually built

Sixteen months, one engineer, forty-three commits. That ratio is the story:
each commit is a shipped capability, not a checkpoint. The repo carries a
329-line roadmap where every feature has a written definition of done, and
twenty-seven of them are closed — each with the commit hash that shipped it
and the measured evidence that proved it.

The discipline started with a decision most AI products skip: before trusting
the pipeline, we built the thing to measure it against. Four public-domain
books — *Alice in Wonderland*, *Hamlet*, a Sherlock Holmes story, *Peter
Rabbit* — were cleaned by hand, paragraph by paragraph, into gold references.
That's why the accuracy numbers in this pitch are percentages with
denominators (124/125 attribution, ≤0.31% contamination), not adjectives.
When the pipeline says 99.2%, it's because a human checked line 125.

Then the gate: after *every* change, the full zero-LLM stress suite must pass
3-for-3 across the corpus before the commit lands. It's written into the
roadmap as a standing rule, the harness lives in the parser itself, and a
dozen archived runs sit in the repo. That gate is the reason a
deterministic-first pipeline stayed deterministic while thirty-three modules
and eighteen thousand lines grew around it — the parser that handled *Peter
Rabbit* in month two still handles it in month sixteen, provably.

The payoff is verifiable today, not promised: feed it *Les Misérables* — the
hardest structural test in public-domain fiction — and it emits 50 parts, 498
chapters, and 723 scenes without a single LLM call. Press one button and a
chaptered, mastered M4B comes out the other end with a per-line timing
manifest.

Everything in this appendix is reproducible from the repository in an
afternoon of diligence: the ledger, the gold corpus, the stress logs, the
rendered masters. We're not asking anyone to trust the demo — we're inviting
them to re-run it.

---

## The 2-minute verbal version

"Every year four million books are published and almost none become audiobooks,
because production costs five to fifteen thousand dollars a title. We built
Firespeaker: upload a manuscript, pick a tier, press one button, and get back a
chaptered audiobook — from a single clean narrator all the way up to a
full-cast, scored, sound-designed audio drama. The analysis pipeline is
deterministic-first with measured 99%+ accuracy against human references, so
authors trust the output; a browser studio lets them correct anything and even
mix Tier 3 scenes on a timeline. Cost of goods is two to eight dollars a book
on rented per-second GPUs — we price at one to three percent of a traditional
production and every tier clears ninety percent gross margin. On the supply
side, performers license their voices — and, as we climb into video, their
likeness and motion — through a consent-based marketplace and get paid per
use; we turn the people AI threatens into our partners. And the same engine that understands a manuscript well enough to
score and cast it is the pre-production layer for video: illustrated editions,
animated episodes, and eventually full screen adaptations ride on top of what
we've already built. The platform is built and rendering books today; we're
raising to license commercial synthesis models, stand up render
infrastructure, and put the first hundred authors through it."
