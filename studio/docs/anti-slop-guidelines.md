# Anti-Slop Guidelines

> Cópia versionada de https://files.barreto.sh/slop.md, que continua sendo a fonte. Está aqui pra que hooks, agentes e CI não dependam de rede pra ler a regra. Ao atualizar o upstream, traga a mudança pra cá no mesmo PR.

A reference for identifying and eliminating AI writing patterns that make text feel artificial, predictable, and low-quality. Covers both English and Brazilian Portuguese, and doubles as the hard-rule source for PRs, commits, code comments, and GitHub review comments.

This file is descriptive: it catalogs AI patterns and shows what natural writing looks like across registers. It does not prescribe a single register. The project style-guide decides the target register (casual blog, technical docs, marketing, etc.). When this file's examples conflict with the project style-guide, the project style-guide wins. For this repo, that's `CLAUDE.md`'s Writing section.

---

## 1. Vocabulary: Words That Scream "AI Wrote This"

### Tier 1: Strongest AI Signals (English)

These words appear 10x-1000x more frequently in LLM output than in human text (per EQ-Bench slop-score analysis and the Antislop framework). Avoid them unless you have a specific reason.

| Category            | Words                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inflated verbs      | delve, embark, underscore, bolster, garner, unpack, harness, navigate, leverage, spearhead, foster, catalyze, propel, utilize, boast, showcase                  |
| Inflated adjectives | crucial, pivotal, multifaceted, nuanced, intricate, meticulous, paramount, indispensable, profound, comprehensive, robust, seamless, vibrant, dynamic, powerful |
| Inflated nouns      | tapestry, landscape, realm, paradigm, synergy, interplay, cornerstone, testament, nexus, bedrock, mosaic, ecosystem                                             |
| Intensifiers        | remarkably, notably, undeniably, unequivocally, profoundly, fundamentally, intrinsically                                                                        |
| Fancy connectors    | furthermore, moreover, additionally, consequently, notwithstanding, henceforth                                                                                  |

### Tier 2: Moderate AI Signals (English)

| Category          | Words                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Marketing verbs   | unlock, empower, transform, revolutionize, streamline, optimize, elevate, amplify                    |
| Vague praise      | innovative, cutting-edge, game-changer, state-of-the-art, groundbreaking, best-in-class              |
| Filler adjectives | valuable, key, significant, essential, vital, important, impactful, compelling, insightful, holistic |
| Drama nouns       | journey, era, dawn, horizon, frontier, odyssey                                                       |

### Tier 1: Strongest AI Signals (Portuguese)

| Category            | Words/Phrases                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Inflated verbs      | fomentar, potencializar, alavancar, concatenar, viabilizar, engendrar, desvelar, perpassar |
| Inflated adjectives | primordial, imprescindivel, inegavel, imensuravel, indubitavel, fulcral                    |
| Fancy connectors    | outrossim, destarte, nesse interim, sob essa otica, nesse diapasao, a luz do exposto       |
| Academic nouns      | paradigma, cerne, bojo, seara, arcabouco, hodierno                                         |

### Tier 2: Moderate AI Signals (Portuguese)

| Category            | Words/Phrases                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Padded verbs        | ressaltar, salientar, elucidar, contemplar, nortear, delinear                                                         |
| Filler adjectives   | robusto, abrangente, assertivo (used incorrectly), holístico, inovador                                                |
| Transition crutches | nesse sentido, diante disso, nesse contexto, sob essa perspectiva, vale ressaltar, cabe destacar, é importante frisar |
| Generic nouns       | cenário, conjuntura, âmbito, esfera, vertente                                                                         |

---

## 2. Phrases: The Dead Giveaways

### Throat-Clearing Openers

These phrases add zero information. Cut them entirely.

**English:**

- "In today's fast-paced world..."
- "In today's rapidly evolving landscape..."
- "In an era where..."
- "It's no secret that..."
- "It goes without saying that..."
- "As we all know..."
- "Let's face it..."
- "In the ever-evolving world of..."

**Portuguese:**

- "No mundo atual..."
- "Nos dias de hoje..."
- "Na sociedade contemporânea..."
- "Desde os primórdios da humanidade..."
- "Ao longo da história..."
- "Vivemos em uma era em que..."
- "É sabido que..."
- "Não é de hoje que..."

**Fix:** Start with the actual point. If the opener can be deleted without losing meaning, delete it.

```
BAD:  "In today's fast-paced world, authentication is crucial for web apps."
GOOD: "Most web apps get authentication wrong."

BAD:  "Nos dias de hoje, a segurança digital é fundamental."
GOOD: "A maioria dos apps web erra na autenticação."
```

### Hedging and Importance Inflation

These phrases pad word count while saying nothing definitive.

**English:**

- "It's important to note that..."
- "It's worth mentioning that..."
- "It should be noted that..."
- "It is widely regarded as..."
- "X is considered to be..."
- "One might argue that..."
- "It is crucial to understand that..."
- "It bears mentioning that..."

**Portuguese:**

- "É importante destacar que..."
- "Vale ressaltar que..."
- "Cabe salientar que..."
- "Torna-se evidente que..."
- "Faz-se necessário..."
- "É fundamental compreender que..."
- "Convém mencionar que..."
- "Cumpre observar que..."

**Fix:** State the thing directly. If it's important, the reader will see that from the content itself.

```
BAD:  "It's important to note that React re-renders the entire subtree."
GOOD: "React re-renders the entire subtree."

BAD:  "Vale ressaltar que o React renderiza toda a subárvore."
GOOD: "O React renderiza toda a subárvore."
```

### Motivational / TED-Talk Closings

**English:**

- "The future is bright."
- "The possibilities are endless."
- "Only time will tell."
- "...and that's a game-changer."
- "The sky's the limit."
- "Embrace the change."
- "The question isn't whether, but when."
- "So, what are you waiting for?"

**Portuguese:**

- "O futuro é promissor."
- "As possibilidades são infinitas."
- "Somente o tempo dirá."
- "Cabe a nós..."
- "O céu é o limite."
- "Precisamos abraçar a mudança."
- "A pergunta não é se, mas quando."
- "Fica a reflexão."

**Fix:** End with something concrete: a specific recommendation, an honest limitation, a next step.

```
BAD:  "The possibilities are endless. Embrace AI and transform your workflow."
GOOD: "Start with one repetitive task. Automate it. See if it saves time. Most don't."

BAD:  "O futuro é promissor. Cabe a nós abraçar essa revolução."
GOOD: "Comece com uma tarefa repetitiva. Automatize. Veja se economiza tempo. Na maioria das vezes, não economiza."
```

### Summative Filler

**English:**

- "In conclusion..."
- "In summary..."
- "Overall..."
- "To sum up..."
- "All in all..."
- "At the end of the day..."
- "The bottom line is..."

**Portuguese:**

- "Em suma..."
- "Em síntese..."
- "Em linhas gerais..."
- "Diante do exposto..."
- "Diante de todo o exposto..."
- "Podemos concluir que..."
- "Frente a isso..."

**Fix:** If your final paragraph just repeats what you said, cut it. A good ending adds a new angle, an honest caveat, or a concrete next step.

---

## 3. Structural Patterns

### The "Not X, But Y" Contrast

The most statistically overused AI construction. The EQ-Bench slop score weights it at 25% of the total score. It appears 6.3x more frequently in LLM output than in human writing.

**Examples:**

- "It's not just a tool — it's a paradigm shift."
- "This isn't about technology. It's about people."
- "It's not a bug, it's a feature of the system."

**Portuguese:**

- "Não se trata apenas de X, mas de Y."
- "Não é apenas uma ferramenta — é uma mudança de paradigma."
- "Não basta X, é preciso Y."

**Fix:** Lead with what it IS. Drop the defensive framing.

```
BAD:  "It's not just a linter. It's a way to enforce team standards."
GOOD: "Biome enforces team standards at the syntax level."

BAD:  "Não se trata apenas de velocidade, mas de experiência do usuário."
GOOD: "Bun é rápido, e isso melhora a experiência do dev."
```

**Sneaky variant: negation-first without the "but".** You don't need "but" in the same sentence for it to be the same vice. Defining something by what it is NOT before what it is (across separate sentences, or by enumerating what got ruled out) is the same move. It slips through because it sounds punchy and grep won't catch it:

```
BAD:  "It wasn't a rounding error. It was three times the budget."
GOOD: "Three times the budget is too much to be a measurement error."

BAD:  "O bug não estava no rádio nem no perfil de advertise; estava num pino de init."
GOOD: "O bug estava num pino de init. Rádio e perfil de advertise estavam limpos."
```

Rule: lead with the affirmative fact. The negation can come after, as context, never as the opener.

### Research Ghosts (Negative Findings)

AI includes things it discovered during research/brainstorming that turned out to be wrong, mentioning them only to negate them. The reader never had that context, so negating something they never heard of is pure noise.

```
BAD:
"Na floresta vivem o morcego Zé e a arara Bela.
Não existe um macaco chamado Jorge nessa região."
(the reader never heard of macaco Jorge, so why bring it up?)

GOOD:
"Na floresta vivem o morcego Zé e a arara Bela."
(only what IS true)
```

**Fix:** The final text should only contain what is true. If something was a dead-end during research, it doesn't appear in the text at all. Write as if you always knew the correct answer.

**Exception: alternatives that were part of the project's story.** A discarded alternative earns its place when weighing it was a real episode in the work: the team considered, discussed, or tested it, and the outcome shaped the design. "We considered a 3.3V clock at reduced duty, tested it, but the vendor wouldn't guarantee it on older meters, so we stepped up to 13.8V" is the story of why the design is what it is. A ghost is different: a path the reader never walked and gains nothing from ("there is no monkey named Jorge"). The test is not whether the alternative had a reason to be dropped (almost all do), but whether the episode mattered enough to the project to be worth telling. Keep those, cut the rest.

### Sentence Stacking

Short, declarative, same-length sentences piled on top of each other with no connective tissue. Each stands alone. None builds on the previous. The reader processes isolated facts instead of following a line of reasoning.

```
BAD:
"Astro is a static site generator. It supports multiple frameworks.
It uses islands architecture. It ships zero JavaScript by default.
It has content collections. It integrates with Tailwind CSS."

GOOD:
"Astro ships zero JavaScript by default, which makes it fast without
trying. When you need interactivity (a search bar, a theme toggle),
you opt in with islands. The rest stays static."
```

### The Rule of Three (Tricolon Abuse)

AI compulsively groups things in threes, even when two or four would be more accurate. Watch for tricolons in headers, bullet points, and descriptive lists.

```
BAD:  "It's fast, flexible, and powerful."
GOOD: "It's fast." (if that's the only thing that matters)

BAD:  "Ele é rápido, flexível e poderoso."
GOOD: "Ele é rápido." (se só isso importa)
```

### Symmetric Sections

AI-generated articles tend to have sections of suspiciously equal length. All 3-4 paragraphs each. All following the same internal structure: topic sentence, elaboration, example, transition. Real articles have sections of wildly different lengths because some ideas need more space than others.

### The Listicle Structure

"7 Ways to X," "5 Tips for Y," "10 Things You Should Know About Z." Every point gets equal treatment. No prioritization. No honest admission that points 8-10 are filler.

**Fix:** If you must list, be opinionated. Say which item matters most. Combine weak points. Cut the ones that don't earn their place.

### The Colon-Header Pattern

AI produces headers with colons constantly:

```
BAD:
"Why Performance Matters: A Deep Dive"
"TypeScript Generics: Understanding the Basics"
"React Server Components: The Future of Web Development"

GOOD:
"Performance matters because users leave"
"TypeScript generics in 10 minutes"
"Server components replace most client-side fetching"
```

### Bold-Word-Colon Inline Pattern

AI loves to write paragraphs as disguised lists:

```
BAD:
**Performance:** Astro ships zero JS. **Flexibility:** Use any framework.
**Developer Experience:** Content collections provide type safety.

GOOD:
Write it as actual prose or an actual list. Don't split the difference.
```

---

## 4. Punctuation Patterns

### Em Dash Overuse

LLMs use em dashes (—) 3-5x more than typical human writing. They substitute em dashes for commas, parentheses, and colons indiscriminately.

```
BAD:  "Astro — a static site generator — ships zero JavaScript — making it fast — by default."
GOOD: "Astro ships zero JavaScript by default, which makes it fast."

BAD:  "O React — uma biblioteca para interfaces — usa um DOM virtual — o que melhora a performance."
GOOD: "O React usa um DOM virtual para melhorar a performance."
```

**Rule:** never use an em dash. Use a comma, a colon, a parenthesis, or start a new sentence instead.

### Semicolon Overuse

AI connects simple phrases with semicolons where conjunctions or periods work better.

```
BAD:  "TypeScript catches errors at compile time; this prevents runtime crashes; developers save debugging time."
GOOD: "TypeScript catches errors at compile time, so you spend less time debugging."
```

### Excessive Formatting

AI text tends to overuse bold, bullet lists, and numbered lists when prose would read better. If you're writing a blog post and more than 30% is bullet points, something is wrong. Reach for a list only when enumerating or comparing three or more genuinely parallel items (steps, files, options); otherwise a paragraph says the same thing more naturally.

### Bold Overuse

AI bolds words for emphasis that prose already carries. A paragraph with three bolded phrases reads like a slide, not writing. Bold is for the rare word a scanning reader must not miss, not for every term you think is important.

```
BAD:  "The **dev server** binds to **0.0.0.0** so it's reachable over the **LAN**."
GOOD: "The dev server binds to 0.0.0.0 so it's reachable over the LAN."
```

**Rule:** if more than one phrase per paragraph is bold, you're decorating, not emphasizing. Delete the bold and let the sentence stand.

### Backtick / Inline-Code Overuse

Backticks are for things you'd type into a machine: commands, file paths, code identifiers, flags, env vars. AI wraps ordinary nouns in backticks to look technical (`performance`, `the team`, `production`). That trains the reader to stop trusting the formatting.

```
BAD:  "The `router` handles `navigation` and keeps the `state` in sync."
GOOD: "The router handles navigation and keeps state in sync."

OK:   "Run `bun run dev`, then edit `src/main.tsx`."
```

**Rule:** if the backticked text isn't something you could paste into a terminal or an editor and have it mean exactly that, drop the backticks.

---

## 5. Tone Patterns

### Fence-Sitting Neutrality

AI avoids taking positions. Everything is "nuanced." Every technology "has its pros and cons." No tool is ever clearly better.

```
BAD:
"Both React and Svelte have their strengths and weaknesses.
The right choice depends on your specific requirements and team expertise.
It's important to evaluate both options carefully."

GOOD:
"Use Svelte for small apps. Use React if you need the ecosystem.
Most teams should pick React because finding developers is easier."
```

**Portuguese:**

```
BAD:
"Tanto o React quanto o Svelte possuem pontos positivos e negativos.
A escolha depende das necessidades específicas de cada projeto.
É fundamental avaliar ambas as opções cuidadosamente."

GOOD:
"Use Svelte pra apps pequenos. Use React se precisa do ecossistema.
A maioria dos times deveria usar React porque é mais fácil achar dev."
```

### Sycophantic Enthusiasm

AI defaults to false excitement. Everything is "exciting," "powerful," "amazing." Real writers express genuine reactions, including skepticism, frustration, or indifference.

```
BAD:  "This exciting new feature is a game-changer for developers!"
GOOD: "This feature replaces a 40-line workaround with a one-liner."

BAD:  "Essa funcionalidade inovadora é um divisor de águas para devs!"
GOOD: "Essa feature substitui um workaround de 40 linhas por uma linha."
```

### Artificial Gravitas

AI inflates the importance of mundane topics by connecting them to sweeping themes.

```
BAD:  "CSS Grid represents a fundamental paradigm shift in how we think about layout on the web."
GOOD: "CSS Grid makes two-dimensional layouts possible without hacks."

BAD:  "O CSS Grid representa uma mudança de paradigma fundamental na forma como pensamos o layout na web."
GOOD: "CSS Grid permite layouts bidimensionais sem gambiarras."
```

### The Empathy Mask

AI pretends to understand feelings it doesn't have. "We've all been there." "I know this can be frustrating." This reads as patronizing when the reader knows it comes from a machine, or from a human copying a machine.

### No Preamble, No Sign-off

AI opens a reply, comment, or PR by acknowledging the request, and closes by inviting more questions. Neither carries information; the model just needs to start with the answer and stop when it's done.

```
BAD:  "Sure, I'd be happy to help with that! Here's how to fix the bug... Let me know if you have any questions!"
GOOD: "The bug is a race condition in the retry loop. Fixed in commit abc123."

BAD:  "Claro, vou te ajudar com isso! Aqui está a correção do bug... Qualquer dúvida, só avisar!"
GOOD: "O bug era uma race condition no loop de retry. Corrigido no commit abc123."
```

**Fix:** say the thing, then stop. Applies to chat replies, PR descriptions, and review comments alike.

---

## 6. Portuguese-Specific Patterns

### Academic Cosplay

LLMs writing in Portuguese default to a register that sounds like a bad academic paper. No real Brazilian writes casually with these constructions:

| AI default                                    | What a person would write     |
| --------------------------------------------- | ----------------------------- |
| "Diante do exposto, torna-se evidente que..." | "Então..." / "Por isso..."    |
| "Faz-se necessário salientar que..."          | (just say the thing)          |
| "No que tange a..."                           | "Sobre..."                    |
| "Em consonância com..."                       | "Como..." / "Igual ao..."     |
| "Sob a égide de..."                           | (never use this in a blog)    |
| "Consoante ao que foi supracitado..."         | "Como eu falei..."            |
| "Destarte..."                                 | "Por isso..."                 |
| "Outrossim..."                                | "Também..." / "Além disso..." |
| "No bojo de..."                               | "Dentro de..."                |
| "A priori..."                                 | "Primeiro..."                 |
| "Hodierno"                                    | "Atual" / "De hoje"           |

### The ENEM Essay Voice

AI Portuguese often sounds like a student writing a vestibular essay: overly formal, full of connectives, no personality. Politicians caught using AI in Brazil were identified by this exact pattern: excessive connectives, predictable block structure, and moralistic conclusions.

**Specific tells:**

- Starting every paragraph with a different connective: "Primeiramente...", "Além disso...", "Nesse sentido...", "Diante disso...", "Portanto..."
- Moralistic closing: "Cabe ao Estado / à sociedade / a nós..."
- Defining a term in the introduction that nobody asked to be defined
- No contractions: "não é" instead of "num é", "para" instead of "pra", "está" instead of "tá"

### Register Mismatch

AI Portuguese defaults to a formal-academic register regardless of context. The fix depends on the target register declared by the project style-guide:

- Casual register (personal blog, informal post): AI strips out contractions, slang, diminutives, and dropped pronouns that a real person would use. Add them back: "pra", "tá", "né", "num"; "rapidinho", "tranquilinho"; "Fiz ontem" instead of "Eu realizei no dia de ontem".
- Formal/technical register (documentation, reports, specs): AI piles on connectives and academic constructions ("destarte", "outrossim", "faz-se necessário", "no bojo de") that no engineer writes. Strip those, but keep the formal base: "para", "está", "não é" stay as-is. The fix is direct prose, not contractions.

The error in both registers is the same: the text doesn't match the register the project demands. Check the style-guide before deciding which side of the fix applies.

### Connector Overload

AI Portuguese piles connectors at the start of every paragraph:

```
BAD:
"Primeiramente, é necessário compreender que...
Além disso, vale ressaltar que...
Nesse sentido, torna-se evidente que...
Diante disso, pode-se afirmar que...
Por fim, é fundamental destacar que..."

GOOD:
"O Astro gera HTML estático por padrão.
Quando você precisa de interatividade (um botão, uma busca),
você opta por uma island. O resto continua estático.
Isso mantém o site rápido sem que você precise pensar em performance."
```

---

## 7. Detection Heuristics

Use these checks to audit your own writing or spot AI-generated text.

### Burstiness Test

Human writing has high burstiness: wild variation in sentence length. AI sentences cluster around 15-20 words.

**Check:** Count the word count of 10 consecutive sentences. If the standard deviation is low (all sentences between 12-22 words), it reads like AI. Human writing ranges from 3-word fragments to 40-word sprawls.

```
AI-like:    14, 17, 15, 18, 16, 14, 17, 15, 19, 16  (low variance)
Human-like: 4, 28, 9, 41, 6, 15, 3, 22, 7, 35       (high variance)
```

### Perplexity Test

AI chooses the most statistically probable next word. Human writing is less predictable. If you can guess the next word in every sentence, the writing has low perplexity.

```
LOW PERPLEXITY (AI-like):
"This comprehensive guide will help you understand the fundamental
concepts of modern web development."
(every word is the most predictable choice)

HIGH PERPLEXITY (human-like):
"I mass-deleted my CSS last Tuesday. The site looked better."
(you couldn't predict any of that)
```

### Paragraph Length Uniformity

Open the text in an editor. Squint. If every paragraph is roughly the same height, it's probably AI. Real writing has one-sentence paragraphs mixed with dense five-sentence paragraphs.

### The "Can I Hear a Voice?" Test

Read the text aloud. Does it sound like a specific person with opinions, or like a press release? AI writing sounds like nobody in particular. Good writing sounds like someone you'd recognize at a bar.

### The Register Test

Match the contractions and formality to the target register declared by the project style-guide. In casual writing, zero contractions is a strong AI signal: humans informally use "don't", "can't", "it's", "won't", "I'll" (or "pra", "pro", "tá", "num", "né" in Portuguese). In formal/technical writing, contractions are not expected; the AI signal there is the opposite: academic-essay phrasing where direct prose belongs ("torna-se imprescindível", "faz-se necessário", "no bojo de").

### The Specificity Test

Does the text contain specific numbers, dates, personal anecdotes, named tools, or concrete measurements? Or is it all abstract claims about "the landscape" and "paradigm shifts"?

```
VAGUE (AI-like):
"Performance is crucial in modern web applications."

SPECIFIC (human-like):
"Our Lighthouse score went from 34 to 97 after removing the carousel."
```

### The Deletion Test

Go through each sentence and ask: "If I delete this, does the text lose meaning?" AI text is full of sentences that can be deleted without anyone noticing. In good writing, every sentence carries weight.

### The First-Sentence Test

AI almost always starts with a broad, generic statement. Humans often start with a specific observation, a question, or a story.

```
AI-like:    "Testing is a crucial part of the software development lifecycle."
Human-like: "I shipped a bug last Friday that tests would have caught in 3 seconds."

AI-like:    "A qualidade de software é um aspecto fundamental no desenvolvimento moderno."
Human-like: "Sexta passada eu subi um bug que um teste pegaria em 3 segundos."
```

---

## 8. What Good Writing Looks Like

Drawn from writers like Paul Graham, Julia Evans, Dan Luu, and Hillel Wayne.

### Paul Graham's Principles

- **Write simply.** "Fancy writing doesn't just conceal ideas. It can also conceal the lack of them."
- **Use examples.** ~70% of his essays contain "for example." Follow abstractions with concrete instances.
- **Delete ruthlessly.** If a sentence is bad, don't fix it. Delete it and try again. Abandon whole paragraphs.
- **Vary sentence length.** Mix short and medium sentences to create conversational rhythm.
- **Be honest.** State what you actually believe, including uncomfortable things.

### Julia Evans's Principles

- **Write about what you struggled with.** Not what you already know: what confused you and how you figured it out.
- **You don't need to be an expert.** Write about what you learned today.
- **Show the journey.** Narrate confusion turning into understanding.
- **Don't dumb it down.** Explain clearly but keep the real technical details.

### Dan Luu's Principles

- **Use examples obsessively.** More examples than you think you need.
- **Be specific.** Name the tools, show the numbers, link the sources.
- **Write plainly.** His blog has zero CSS (practically). The writing does the work.
- **Find your own voice.** Don't copy anyone's style. Find what works for you.

### Hillel Wayne's Principles

- **Make complex things accessible.** Use practical applications, not abstract theory.
- **Stake a claim.** Take positions. Argue for them. Be willing to be wrong.
- **Connect to real consequences.** Show why the technical detail matters in practice.

### The Common Thread

Every good technical writer shares these traits:

1. **Specificity over abstraction.** Named tools, real numbers, actual experiences.
2. **Opinions over neutrality.** "X is better" not "X has its strengths."
3. **Short sentences alongside long ones.** Rhythm, not uniformity.
4. **Deletion over addition.** Cut the sentence that doesn't earn its place.
5. **Honest uncertainty.** "I don't know" beats "further research is needed."

---

## 9. Before/After Examples

### Example 1: Blog Post Introduction

**AI slop (English):**

> In today's rapidly evolving technological landscape, web performance has become a crucial consideration for developers. It's important to note that user experience is fundamentally shaped by how quickly a page loads. In this comprehensive guide, we'll delve into the intricacies of Core Web Vitals and explore how they can transform your development workflow.

**Human version:**

> Our site took 8 seconds to load on 3G. Half our users are on 3G. We were losing them before the page even rendered. Here's what we changed.

**AI slop (Portuguese):**

> No cenário atual do desenvolvimento web, a performance tornou-se um aspecto fundamental e imprescindível. Nesse sentido, vale ressaltar que a experiência do usuário é profundamente impactada pelo tempo de carregamento. Diante disso, este guia abrangente se propõe a elucidar as nuances dos Core Web Vitals e como eles podem transformar seu fluxo de trabalho.

**Human version:**

> Nosso site levava 8 segundos pra carregar em 3G. Metade dos usuários tá em 3G. Eles iam embora antes de ver qualquer coisa. Aqui o que a gente mudou.

### Example 2: Technical Explanation

**AI slop:**

> TypeScript's type system is a powerful and multifaceted tool that enables developers to write more robust and maintainable code. By leveraging the comprehensive type inference capabilities, teams can significantly reduce the occurrence of runtime errors, ultimately leading to a more seamless development experience.

**Human version:**

> TypeScript catches a class of bugs that would otherwise blow up in production. The compiler yells at you before your users do. That's the whole value proposition.

### Example 3: Conclusion

**AI slop:**

> In conclusion, Astro represents a paradigm shift in how we approach web development. By embracing its innovative islands architecture, developers can unlock unprecedented levels of performance. The future of web development is bright, and Astro is leading the way. The possibilities are truly endless.

**Human version:**

> Astro does one thing well: it ships less JavaScript. For content sites, that's enough. For apps with heavy interactivity, you'll fight it. Pick accordingly.

**Portuguese version:**

> O Astro faz uma coisa bem: manda menos JavaScript pro navegador. Pra sites de conteúdo, isso basta. Pra apps com muita interatividade, você vai brigar com ele. Escolha de acordo.

### Example 4: Describing a Tool

**AI slop:**

> Biome is a comprehensive, cutting-edge toolchain that seamlessly integrates linting and formatting into a unified developer experience. It's not just a linter — it's a paradigm shift in how teams approach code quality.

**Human version:**

> Biome does what ESLint + Prettier do, but in one tool and 100x faster. We switched last month. Setup took 20 minutes.

---

## 10. Quick Reference Checklist

Before publishing, scan your text for:

- [ ] **Throat-clearing opener?** Delete the first paragraph and see if the second one is a better start.
- [ ] **Hedge phrases?** Search for "important to note", "worth mentioning", "vale ressaltar", "cabe destacar". Delete all of them.
- [ ] **"Not X, but Y" patterns?** Count them. More than one per piece is a red flag. Rewrite to lead with Y.
- [ ] **Tricolons?** Are things grouped in threes artificially? Drop the weakest item.
- [ ] **Any em dash?** Zero tolerance now. Rewrite as a comma, a colon, a parenthesis, or a new sentence.
- [ ] **Semicolon count?** More than 1 per 500 words in casual writing? Use periods.
- [ ] **Sentence length variety?** Read aloud. Does it feel monotonous? Add a 4-word sentence. Break up a long one.
- [ ] **Motivational ending?** Replace with a concrete recommendation or honest caveat.
- [ ] **Register matches the project style-guide?** Casual contexts need contractions and natural rhythm; formal/technical contexts stay formal but avoid academic-essay constructions. Check the style-guide before deciding which side to lean.
- [ ] **Specific details?** Does the text contain at least one number, date, or named tool per section?
- [ ] **Voice?** Could this have been written by anyone? If yes, it needs an opinion, a story, or a specific experience.
- [ ] **Connector pile-up (PT)?** Do 3+ consecutive paragraphs start with connectives? Rewrite them.
- [ ] **Academic cosplay (PT)?** Replace "faz-se necessário" with the direct statement. Replace "destarte" with "por isso."
- [ ] **Bold-colon pattern?** If you have `**Word:** explanation` more than twice in a row, make it a real list or real prose.
- [ ] **Bold overuse?** More than one bold phrase per paragraph means you're decorating. Delete the bold.
- [ ] **Backtick overuse?** If the backticked text isn't a command, path, identifier, flag, or env var you could paste into a terminal or editor, drop the backticks.
- [ ] **Preamble or sign-off?** A chat reply, PR, or comment that opens with an acknowledgment or closes by inviting questions loses both. Start with the answer.
- [ ] **Tool-attribution footer?** No "Generated with Claude Code," "Co-Authored-By: Claude," or robot emoji in any commit, PR, or comment.
- [ ] **Paragraph as one line in source?** Each paragraph is a single continuous line in the markdown source, not hard-wrapped at a column count.
- [ ] **Can you delete a paragraph and nobody notices?** Delete it.

---

## 11. PRs, Commits, and Review Comments

Rules specific to code-review workflows: PR descriptions, commit messages, inline code comments, and GitHub comments. These apply whether a person or an agent writes the text.

### Hard Rules

- Never add "Generated with Claude Code," "Co-Authored-By: Claude," or any tool-attribution or robot-emoji footer, in a commit, a PR, or a comment.
- No emoji stuffing. A checklist marker is fine; decorative emoji are not.
- No marketing tone and no self-congratulation ("comprehensive," "robust," "seamlessly"), and no filler that just restates the title.

### PR Descriptions

- Lead with what changed and why, in plain language. Assume a reviewer who already knows the codebase.
- Prefer short sections: what changed, how, acceptance criteria, verification.
- State design decisions and trade-offs explicitly, especially anything the reviewer would otherwise have to reconstruct.
- Tie back to the issue ("Closes #N") and mirror its acceptance criteria.
- Verification means the exact commands run and their result (`bun run doctor`, `bun run test:e2e`), not a vague claim of "tested."
- Keep it as long as it needs to be, no longer. Cut anything a reviewer would skip.

### Code Comments

Be sparing with single-line `//` comments. Comment a non-obvious WHY, never narrate what the code already says. Match the file's existing comment density.

### GitHub Comments

Be direct and specific. Reference `file:line` instead of describing a location in prose. When reporting a result, give the evidence: output, link, or status, not a summary of what should have happened.

---

## Sources

- [Wikipedia: Signs of AI Writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- [EQ-Bench Slop Score](https://eqbench.com/slop-score.html)
- [Antislop: A Framework for Identifying LLM Patterns (ICLR 2026)](https://arxiv.org/abs/2510.15061)
- [The Field Guide to AI Slop (Charlie Guo)](https://www.ignorance.ai/p/the-field-guide-to-ai-slop)
- [Don't Write Like AI (Blake Stockton)](https://www.blakestockton.com/takeaways-from-wikipedias-signs-of-ai-writing-2/)
- [Why ChatGPT Writes Like That (Colin Gorrie)](https://www.deadlanguagesociety.com/p/rhetorical-analysis-ai)
- [Delving into LLM-assisted Writing (Science Advances)](https://www.science.org/doi/10.1126/sciadv.adt3813)
- [GitHub: llm-cliches](https://github.com/nanxstats/llm-cliches)
- [GitHub: stop-slop](https://github.com/hardikpandya/stop-slop)
- [GitHub: humanize-writing-skill](https://github.com/lguz/humanize-writing-skill)
- [Recognizing AI Structures (Kassorla & Novokshanova)](https://michellekassorla.substack.com/p/recognizing-ai-structures-in-writing)
- [Common Sentence Structures in AI Writing (Stryng)](https://stryng.io/common-sentence-structures-in-ai-generated-text/)
- [Cliches de IA invadem textos de politicos brasileiros (Gazeta do Povo)](https://www.gazetadopovo.com.br/vida-e-cidadania/cliches-de-ia-invadem-textos-de-politicos-brasileiros-nas-redes/)
- [Texto Gerado por IA em Portugues (Hastewire)](https://hastewire.com/pt/blog/texto-gerado-por-ia-em-portugues-como-identificar-facilmente)
- [Densidade lexical em textos gerados pelo ChatGPT (SciELO)](https://www.scielo.br/j/tl/a/crx3yywCw3LSxtjtdv44mDC/?format=html&lang=pt)
- [Write Simply (Paul Graham)](https://paulgraham.com/simply.html)
- [Blog About What You've Struggled With (Julia Evans)](https://jvns.ca/blog/2021/05/24/blog-about-what-you-ve-struggled-with/)
- [Some Thoughts on Writing (Dan Luu)](https://danluu.com/writing-non-advice/)
- [AI Slop, Suspicion, and Writing Back (Ben Congdon)](https://benjamincongdon.me/blog/2025/01/25/AI-Slop-Suspicion-and-Writing-Back/)
- [Perplexity and Burstiness in AI vs. Human Writing](https://originality.ai/blog/perplexity-and-burstiness-in-writing)
- [300+ AI Words to Avoid (ContentBeta)](https://www.contentbeta.com/blog/list-of-words-overused-by-ai/)
