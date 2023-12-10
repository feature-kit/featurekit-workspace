<script>
    function clamp(num, min, max) {
		return num < min ? min : num > max ? max : num;
	}

    import WeightedDoc from './WeightedDoc.svelte';

    // 2d array of tokens
    export let tokens;
    // 2d array of weights
    export let weights;
    export let reversed = false;
    // value to sort by per doc
    // export let per_doc_vals;

    let decor = (v, i) => [v, i];          // set index to value
    let undecor = a => a[1];               // leave only index
    let argsort = arr => arr.map(decor).sort().map(undecor);


    let per_doc_vals = []

    for (let i = 0; i < weights.length; i++) {
        if (!reversed) {
            per_doc_vals.push(Math.max(...weights[i]))
        } else {
            per_doc_vals.push(Math.min(...weights[i]))
        }
    }
    console.log(per_doc_vals)
    let perm = argsort(per_doc_vals);
    console.log(perm)
    if (reversed) {
        console.log(perm[0])
        perm.reverse();
        console.log('reversed')
        console.log(perm[0])
        console.log(perm[1])
    }
    // perm = perm.reversed()
    

    function permute(arr, perm) {
        let result = []
        for (let i = 0; i < perm.length; i++) {
            result.push(arr[perm[i]])
        }
        return result
    }

    tokens = permute(tokens, perm)
    weights = permute(weights, perm)
    per_doc_vals = permute(per_doc_vals, perm)


    let renderedDocs = [];
    let renderedWeights = [];

    let start = 1.0;
    if (reversed) {
        start = -1.0
    }
    let end = 1.0;

    function shuffle(array) {
        let currentIndex = array.length,  randomIndex;

        // While there remain elements to shuffle.
        while (currentIndex > 0) {

            // Pick a remaining element.
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;

            // And swap it with the current element.
            [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
        }

        return array;
    }

    

    function updatePercentiles(newStart, newEnd) {
        console.log('updating percentiles')


        let startIdx = clamp(tokens.length - 1 - 20, 0, tokens.length-1);
        let endIdx = tokens.length-1;

        for (let i = 0; i < tokens.length-20; i++) {

            if (!reversed && per_doc_vals[i] > start ) {
                startIdx = i;
                break;
            }
            if (reversed && per_doc_vals[i] < start) {
                startIdx =i;
                break;
            }
        }

        let tempDocs = []
        let tempWeights = []

        let stepSize = clamp(Math.round((endIdx-startIdx)/20), 1, endIdx-startIdx)

        for (let i = startIdx; i < endIdx; i += stepSize) {
            tempDocs.push(tokens[i])
            tempWeights.push(weights[i])
        }

        // shuffle the arrays
        let perm = []
        for (let i = 0; i < tempDocs.length; i++) {
            perm.push(i)
        }
        perm = shuffle(perm)
        tempDocs = permute(tempDocs, perm)
        tempWeights = permute(tempWeights, perm)

        renderedDocs = [...tempDocs]
        renderedWeights = [...tempWeights]

    }

    function handleInputChange(event) {
        start = event.target.value;
    }

    function handleChange() {
        updatePercentiles(start, end);
        console.log(renderedDocs[0])
        console.log('should be done')
    }
    
    updatePercentiles(start, end);
</script>


<div class="weighted-docs">
    <div class="slider">
        <div>
            <input type="range" min="-1" max="1" step="0.02" bind:value={start} on:input={handleInputChange} on:change={handleChange}/>{start}
        </div>
    </div>
    
    {#each renderedDocs as doc, i}
        <div class="doc">
            <WeightedDoc tokens={doc} weights={renderedWeights[i]} />
        </div>
    {/each}
</div>

<style>
    .weighted-docs {
        background: #1b1a1a;
        color: rgb(197, 193, 187);
        padding: 2rem;
    }
    .slider{
        padding: 2rem;
        width: 60%;
	}
    .doc {
        margin: 1rem;
    }
</style>