

<script>
    import DoubleSlider from './DoubleSlider.svelte';
	import LabeledSelector from './LabeledSelector.svelte';

    export let thresholdMin = -1;
    export let thresholdMax = 1;
    export let start = thresholdMin;
    export let end = thresholdMax;

    export let thresholdOrPercentile = "threshold";
	export let ordering = 'descending';
	export let aggregation = 'max';

	export let resampleDocs;
</script>


<main>

<div class='menu'>
	<DoubleSlider min={(thresholdOrPercentile == 'percentile') ? 0 : thresholdMin} max={(thresholdOrPercentile == 'percentile') ? 1 : thresholdMax} bind:start bind:end/>
	<div class='interface'>
		<LabeledSelector options={['max', 'mean', 'min']} title='aggregation' bind:selection={aggregation}/>
		<LabeledSelector options={['threshold', 'percentile']} title='slider type' bind:selection={thresholdOrPercentile}/>
		<LabeledSelector options={['descend','ascend', 'random']} title='sort' bind:selection={ordering}/>
		<div>
		<button on:click={resampleDocs}>sample</button>
		</div>
	</div>
</div>

</main>



<style>
	.interface {
		display: flex;
	}
	.menu {
		display: flex;
		height: 100%;
	}

	button {
		height: 40%;
		margin: 0.45rem;
		border-radius: 0.5rem;
		border: 1px solid;
		margin-left: 0.3rem;
		margin-right: 0.3rem;
		padding: 0.2rem;
		/* margin-top: auto; */
		/* bottom: 0; */
		/* width: 4.5rem; */
	}


	main {
		font-family: Sans-Serif;
		width: min(100%, 40rem);
		background-color: rgb(197, 197, 197);
		padding: 1rem;
		border-radius: 10px;
	}
</style>