<script>
    function clamp(num, min, max) {
		return num < min ? min : num > max ? max : num;
	}

    import WeightedDoc from './WeightedDoc.svelte';

    // 2d array of tokens
    export let tokens;
    // 2d array of weights
    export let weights;
    // value to sort by per doc
    export let per_doc_maxes;

    let renderedDocs = [];
    let renderedWeights = [];

    // import DoubleRangeSlider from './DoubleRangeSlider.svelte';
    let start = 1.0;
    let end = 1.0;
    const nice = d => {
		if (!d && d !== 0) return '';
		return d.toFixed(2);
	}

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

    function permute(arr, perm) {
        let result = []
        for (let i = 0; i < perm.length; i++) {
            result.push(arr[perm[i]])
        }
        return result
    }

    function updatePercentiles(newStart, newEnd) {
        console.log('updating percentiles')


        let startIdx = clamp(tokens.length - 1 - 20, 0, tokens.length-1);
        let endIdx = tokens.length-1;

        for (let i = 0; i < tokens.length-20; i++) {
            if (per_doc_maxes[i] > start) {
                startIdx = i;
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
            <input type="range" min="0" max="1" step="0.01" bind:value={start} on:input={handleInputChange} on:change={handleChange}/>
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