<script>
    import WeightedDoc from './WeightedDoc.svelte'
    import WeightedDocsControl from './components/WeightedDocsControl.svelte'
	
    // import docs from './docs.json';
    // import acts from './feature_acts.json';
   
    export let docs;
    export let acts;
    

    let decor = (v, i) => [v, i];          // set index to value
    let undecor = a => a[1];               // leave only index
    let argsort = arr => arr.map(decor).sort().map(undecor);
    
    let aggrs = {}
    let aggrPerms = {}


    aggrs['max'] = acts.map((feats) => Math.max(...feats))
    aggrPerms['max'] = argsort(aggrs['max'])
    aggrs['max'] = aggrs['max'].toSorted()

    aggrs['min'] = acts.map((feats) => Math.min(...feats))
    aggrPerms['min'] = argsort(aggrs['min'])
    aggrs['min'] = aggrs['min'].toSorted()

    aggrs['mean'] = acts.map((feats) => feats.reduce((a, b) => a + b, 0) / feats.length)
    aggrPerms['mean'] = argsort(aggrs['mean'])
    aggrs['mean'] = aggrs['mean'].toSorted()

    // let docs_ = docs.slice(0,20)
    // let acts_ = acts.slice(0,20)
    // let aggrMax = aggrMax.slice(0,20)

    let aggr = 'max'
    let thresholdOrPercentile = 'threshold'
    let ordering = 'descend';

    

    // let aggrMaxPerm = argsort(aggrMax)
    // let aggrMaxSorted = aggrMax.toSorted()

    function getRandomInts(rangeStart, rangeEnd, count) {
        // Create an array with numbers from rangeStart to rangeEnd
        let numbers = [];
        for (let i = rangeStart; i < rangeEnd; i++) {
            numbers.push(i);
        }

        // Shuffle the array
        for (let i = numbers.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [numbers[i], numbers[j]] = [numbers[j], numbers[i]]; // Swap elements
        }

        // Return the first 'count' elements
        return numbers.slice(0, count);
    }

    // export let thresholdMin = Math.min(...acts.map(r=>Math.min(...r)))-1
    // export let thresholdMax = Math.max(...acts.map(r=>Math.max(...r)))*2
    let thresholdMin=0
    let thresholdMax=1

    export let start = thresholdMin;
    export let end = thresholdMax;

    export let k = 10;

    let getDocBounds = () => {
        let startBound, endBound;
        if (thresholdOrPercentile == 'threshold') {
            for (var i = 0; i <= Math.max(docs.length-k, 0); i++) {
                if (aggrs[aggr][i] >= start) {
                    break; 
                }
            }
            startBound = i;

            for (var j = aggrs[aggr].length; j >= startBound+k; j--) {
                if (aggrs[aggr][j-1] <= end && aggrs[aggr][j-1] >= start) {
                    break;
                }
            }
            endBound = j;
        } else {
            // percentile
            startBound = Math.min(Math.floor(aggrs[aggr].length * start), aggrs[aggr].length-k);
            endBound = Math.max(Math.ceil(aggrs[aggr].length * end), k)
            console.log(startBound, endBound)
        }
        return [startBound, endBound];
    };
    

    let getDocIndices = () => {
        let [startBound, endBound] = getDocBounds()
        console.log([startBound, endBound])
        let indices = getRandomInts(startBound, endBound, k)
        if (ordering == 'ascend') {
            indices.sort()
        } else if (ordering == 'descend') {
            indices.sort((a,b) => b-a)
        } else {
            // nothing
        }
        return indices.map(i => aggrPerms[aggr][i])
    }

    let renderedDocIndices;
    let resampleDocs = () => {
        if (thresholdOrPercentile == 'percentile') {
            start = Math.max(start, 0)
            end = Math.min(1, end)
        }
        renderedDocIndices = getDocIndices()
        console.log(renderedDocIndices)
    }

    
    $: {
        thresholdOrPercentile; start; end; ordering;
        resampleDocs()
    }

    $: {
        console.log('tick')
        thresholdMin = aggrs[aggr][0]
        {console.log(docs)}
        thresholdMax = aggrs[aggr][docs.length - 1]
        console.log(thresholdMin)
    }
</script>

<main>
    <WeightedDocsControl thresholdMax={thresholdMax} thresholdMin={thresholdMin} bind:start={start} bind:end={end} bind:thresholdOrPercentile={thresholdOrPercentile} bind:aggregation={aggr} bind:ordering={ordering} resampleDocs={resampleDocs}/>
        {#each renderedDocIndices as i}
            <WeightedDoc tokens={docs[i]} weights={acts[i]}/>
        {/each}
</main>

<style>
    main {
        background-color: black;
    }
</style>