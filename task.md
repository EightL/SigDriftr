
3

Automatic Zoom
Media Intelligence Hackathon
Lakmoos ·March 19–20, 2026
Overview
The task is to build a service that collects media content from selected sources and produces useful
summaries and insights about what is happening around the world. The focus is on news analysis, not
UI.
Your service should be able to answer questions such as:
• What are the main technology stories today?
• What topics dominate the media in a specific country?
• How does sentiment differ between sources?
How you solve this is largely up to you.
Media Sources
To keep the scope focused and comparable across teams, the hackathon is limited to a small set of
countries and media categories.
Required Sources
Each team must include at least one source from each of the following:
• Czech Republic (CZ)
• Germany (DE)
• International / World News (e.g., Reuters, AP, BBC)
Sources can be accessed via RSS feeds, public APIs, or scraping, depending on availability.
Example: developer.nytimes.com/apis
Optional Extensions
Teams may additionally include sources from:
• United Kingdom (UK)
• Italy (IT)
• Spain (ES)
• France (FR)
No other countries or categories are expected to be part of this challenge.
Core Requirement
Your service must expose at least one HTTP endpoint that returns media summaries, filtered by:
• Topic — e.g. technology, medicine, politics, ...
• Country — e.g. CZ, DE, global
• Source — specific outlet or all sources
• ...
The exact response format, internal data model, and storage approach are not prescribed and are up to
you.
Evaluation Criteria
The projects will be evaluated on the following dimensions:
1. Quality and clarity of summaries
2. Relevance of extracted insights
3. Usefulness of filtering by topic, country, and source
4. Soundness of technical decisions
5. Potential to evolve into a larger system
What We Care About
• Ability to distinguish signal from noise
• Reasonable handling of importance and relevance
• Thoughtful use of NLP or ML techniques
• Clear technical reasoning behind design decisions
What We Do Not Require
• A frontend application (though a basic UI is highly encouraged and would be a great addition, it is not
a formal requirement)
• Authentication or user management
• Production-ready infrastructure
Repository and Technical Evaluation
In addition to the live demo, judges will also review the submitted GitHub Classroom repository. The
goal is not only to assess what the system does, but also how the solution is structured and implemented.
In particular, we will consider:
• Code quality — readability, maintainability, and reasonable use of language features, libraries, and
abstractions
• Architecture and extensibility — whether the solution could realistically evolve into a larger system
• Quality of sources — whether the selected media sources are relevant and appropriate
• Quality of data collection — robustness and thoughtfulness of scraping, APIs, parsing, and ingestion
• Quality of processing — how well the collected data is cleaned, filtered, enriched, and prepared for
analysis
• Quality of analysis and output — usefulness, relevance, and clarity of the produced summaries and
insights
• Documentation — quality of the README, setup instructions, and helpful code comments
• Bonus task submission — if completed, it may positively affect the final evaluation
These criteria are intended as guidance rather than a strict checklist. Judging will combine this technical
review with the overall usefulness and quality of the demonstrated solution.
Technical Stack
The required programming language for this hackathon is Python. Beyond that, you are free to choose:
• Web framework (e.g. FastAPI, Flask, Django)
• Database or storage solution
• NLP approach: rules, classical NLP, embeddings, LLMs (a GPT API key will be provided), or a mix
• Update frequency and processing strategy
• Any Python libraries or tools you find useful
Simple and well-justified solutions are preferred over complex ones.
Deliverables
By the end of the hackathon, each team should provide:
• A running backend service or working demo, preferably Dockerized
• Example API requests and responses
• A short live demo and presentation of the final solution
Submissions
Each team will receive access to a shared repository through GitHub Classroom. To join, use the following
invitation link:

This repository is the official submission location for the project. All work related to the hackathon should
be committed to your team’s repository. Your final solution must be pushed before the submission deadline.
Unless stated otherwise, the version considered submitted will be the state of the main branch in the
GitHub Classroom repository at the submission deadline. Teams may optionally make the final version
easier to identify by:
• creating a Git tag tag such as final-submission, or
• using a clearly marked final commit message such as FINAL SUBMISSION
This helps ensure that the evaluated version matches the version presented during the demo.
Each submission should include:
• Source code of the project
• README with clear instructions on how to run the project locally
• Example API requests and responses
• Presentation slides summarizing the approach and results
If your project depends on additional services (e.g. a database), a Dockerized setup is strongly encour-
aged.
Presentations
Each team has 10 minutes to present their solution, followed by a short Q&A. The presentation should
ideally include:
• The architecture of the solution (data collection, processing, analysis)
• The most interesting or innovative parts of the implementation
• Challenges encountered during development and how they were addressed
• A short demo of the system or API
