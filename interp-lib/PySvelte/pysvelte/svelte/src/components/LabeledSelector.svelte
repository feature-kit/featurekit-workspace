<script>
    export let options;
    export let title;
    export let selection;

    let titleClass = 'title'

    if (title === '') {
        titleClass = 'hiddenTitle'
        title = 'hidden'
    }
</script>


<main>

<div class={titleClass}>
{title}
</div>

<div>
<select class='select' name={title} bind:value={selection}>
    {#each options as option}
        <option class='option' value={option}>{option}</option>
    {/each}
</select>
</div>
</main>


<style>
    .title {
        justify-content: center;
        align-items: center;
        font-size: .8rem;
        color: rgb(45, 45, 45);
    }
    .hiddenTitle {
        font-size: .8rem;
        color: rgb(45, 45, 45);
        opacity: 0;

    }
    main {
        padding: .1rem;
        flex-shrink:1;
    }

    .option {
        width: 5rem;
    }

    select {
        width: 4.9rem;
    }
</style>