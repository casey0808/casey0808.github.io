1. **Reducer** reduce a set of actions (over time) into a single state.

**Important Rule of Reducer:**
#1 Never return undefined from a reducer.

2. **Actions** are very free-formed things, as long as it's an object with a type.
convention: types are **plain strings**, and often uppercased.