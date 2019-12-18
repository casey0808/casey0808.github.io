1. **Reducer** reduce a set of actions (over time) into a single state.

2. **Important Rule of Reducer:**
#1 Never return undefined from a reducer.

3. **Actions** are very free-formed things, as long as it's an object with a type.

··· *convention*: types are **plain strings**, and often uppercased.
Actions don't really do anything on their own. In order to make an action DO something, you need to **dispatch** it.

4. 