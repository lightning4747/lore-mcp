1. **Architecture & Boundaries:** Follow Hexagonal/Clean Architecture (`domain/` -> `application/` -> `infrastructure/`). Dependencies MUST point inward. `domain/` depends on NOTHING external.
2. **Abstractions First:** Define domain interfaces before writing concrete logic. External APIs, databases, and third-party services MUST use the Adapter pattern wrapped around an application interface.
3. **Type Safety & Design:** Use branded/nominal types for domain IDs (`type UserId = string & { readonly __brand: unique symbol }`) to prevent primitive obsession. Favor high cohesion and loose coupling.
4. **Implementation Discipline:** Work in tiny, single-purpose increments. Never implement full features in one pass.
5. **Quality Gate:** Code must compile cleanly with zero implicit type casting (`any`) and pass all architectural boundaries before completing a task.
6. Only do the task i say, nothing more, nothing less
7. After completing each logical unit of work, verify it, then create a Git commit with a concise message.
