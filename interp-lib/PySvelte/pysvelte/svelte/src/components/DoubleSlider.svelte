<script>
    import DoubleRangePercentSlider from './DoubleRangePercentSlider.svelte';
    export let min = -1;
    export let max = 1;

    export let start = min;
    export let end = max;

    let zeroPercent;
    zeroPercent = -100*min/(max-min)
    let percentile_start=0
    let percentile_end=1
    const nice = d => {
		if (!d && d !== 0) return '';
		return d.toFixed(2);
	}

    $: start = percentile_start*(max-min)+min
    $: end = percentile_end*(max-min)+min


  </script>

<main>

    <div class="slider-and-marker">
        <DoubleRangePercentSlider bind:start={percentile_start} bind:end={percentile_end} />
        {#if ((max > 0) && (min < 0))}
            <div class="zero-marker" style="left: {zeroPercent}%;"></div>
        {/if}
    </div>
    <div class="labels">
        <div class="label">{nice(start)}</div>
        <div class="label">{nice(end)}</div>
    </div>
    
    
</main>


<style>
	main {
		text-align: center;
		padding: 0.2rem;
		max-width: 40rem;
		margin: 0 auto;
        margin: .9rem;
	}
	.label:first-child {
		float: left;
        color: black;
	}
	.label:last-child {
		float: right;
        color: black;
	}
	.slider-and-marker{
        position: relative;
	}
    .zero-marker {
        position: absolute;
        height: 100%; /* Adjust as needed */
        width: .15rem; /* Width of the line */
        background-color: black; /* Color of the line */
        top: 0;
        bottom: 0;
        pointer-events: none;
    }

    main {
        flex: 1;
        font-family: sans-serif;
    }

</style>