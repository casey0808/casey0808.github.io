* **Reducer** reduce a set of actions (over time) into a single state.

* **Important Rule of Reducer:**
    * Never return undefined from a reducer.
    * Reducers must be pure functions. 

* **Actions** are very free-formed things, as long as it's an object with a type.
> *convention*: types are **plain strings**, and often uppercased.

* Actions don't really do anything on their own. In order to make an action DO something, you need to **dispatch** it.

* Call **dispatch** with an action, and Redux will call reducer with that action (and then replace the state with whatever the reducer returns).