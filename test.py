from GraphFlow.wiseragents_creation import graph

# Input
topic = "The benefits of adopting Scrumban in software development"
thread = {"configurable": {"thread_id": "1"}}

# before-feedback WiserAgents (until the first interruption)
for event in graph.stream({"topic":topic}, thread, stream_mode="values"):
    # Review
    wiseragents = event.get('wiseragents', '')
    if wiseragents:
        for wiseragent in wiseragents:
            print(f"Name: {wiseragent.name}")
            print(f"Domain Expertise: {wiseragent.domain_expertise}")
            print(f"Description: {wiseragent.description}")
            print(f"Wisdom Score: {wiseragent.WS}")
            print(f"Preferred LLM: {wiseragent.preferred_llm}")
            print("-" * 50)  

# Get state and look at next node
state = graph.get_state(thread)
print(state.next)

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"human_wiseragent_feedback": 
                            "Add in a WiserAgent that...(e.g: has legal and authority domain speciality)"}, as_node="human_feedback")

# After-feedback WiserAgents
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    wiseragents = event.get('wiseragents', '')
    if wiseragents:
        for wiseragent in wiseragents:
            print(f"Name: {wiseragent.name}")
            print(f"Domain Expertise: {wiseragent.domain_expertise}")
            print(f"Description: {wiseragent.description}")
            print(f"Wisdom Score: {wiseragent.WS}")
            print("-" * 50) 

# if satisfied, we simply supply no feedback
further_feedack = None
graph.update_state(thread, {"human_wiseragent_feedback": further_feedack}, as_node="human_feedback")

# Continue the graph execution to end
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Final WiserAgents--")
    node_name = next(iter(event.keys()))
    print(node_name)
print("final_state :", graph.get_state(thread))
Final_wiseragents = graph.get_state(thread).values.get('wiseragents')
print("wiseragents :", Final_wiseragents)

