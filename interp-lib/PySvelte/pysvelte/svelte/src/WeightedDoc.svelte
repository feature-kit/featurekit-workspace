<script>
    function zip(...arrays) {
        const length = Math.min(...arrays.map(arr => arr.length));
        return Array.from({ length }, (_, i) => arrays.map(arr => arr[i]));
    }
    export let title = '';
    export let tokens;
    export let weights;

    const nice = d => {
		if (!d && d !== 0) return '';
		return d.toFixed(2);
	}

    let zipped = zip(tokens, weights)

    function getStyle(weight) {
        if (weight >= 0) {
            return `background-color:rgba(13,220,193,${weight});`
        } else {
            return `background-color:rgba(220,13,48,${-weight});`
        }
    }

    $: zipped = zip(tokens, weights)
</script>

<h3 class="title">{title}</h3>
<div>
    
    {#each zipped as [tok, weight]}
        <span class="token" style="{getStyle(weight)}">{tok}
            <span class="hovertext">{nice(weight)}</span>
        </span>
    {/each}
</div>

<style>
    .title {
        /* font-size: 1rem; */
        /* padding: 0rem; */
    }
    .token {
        border:0px solid black;
        padding:0px;
        font-size: larger;
        position: relative;
        cursor: default;
        margin-left: -0.15rem;
        margin-left: -0.15rem;
        color: white;
    }
    .token:hover {
        background-color:rgba(135,206,250,1);
    }
    .token .hovertext {
        display: none;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -100%);
        z-index: 1;
        background: #f6e0e0;
        color: black;
        margin-top: -.5rem;
        padding: .2rem;
    }

    .token:hover .hovertext {
        visibility: visible;
        display: block;
    }
</style>