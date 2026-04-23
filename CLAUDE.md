# Development workflow

- Active development lands on the `coqui-dev` branch. New commits go here.
- The user (@TheNewJavaman) reviews `coqui-dev` on GitHub via PR before merging
  into `coqui`. `coqui` is the integration/stable branch; never commit to it
  directly — only fast-forward merge via reviewed PR.
- `main` and `stable` track upstream AFL++ — never push there, never PR to
  them from this project.
- Remote `origin` is `git@github.com:TheNewJavaman/AFLplusplus.git`; remote
  `upstream` is the real AFL++ repo and should not be pushed to.
