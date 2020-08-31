t_add_node = 0
t_copy_vars = 0
t_copy_constraints = 0
t_copy_nl_constraints = 0
using JuMP
#############################################################################################
# Combine: IDEA: Group nodes together into a larger node
#############################################################################################

abstract type AbstractLinkConstraint <: JuMP.AbstractConstraint end

struct LinkConstraint{F <: JuMP.AbstractJuMPScalar,S <: MOI.AbstractScalarSet} <: AbstractLinkConstraint
    func::F
    set::S
end

LinkConstraint(con::JuMP.ScalarConstraint) = LinkConstraint(con.func,con.set)

mutable struct CombinedNode
    index::Int64
    obj_dict::Dict{Symbol,Any}
    variablemap::Dict{JuMP.VariableRef,JuMP.VariableRef}                    #map from combine model variable to original modelgraph variable
    constraintmap::Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}
    nl_constraintmap::Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}
    objective::Union{JuMP.AbstractJuMPScalar,Expr}                          #copy of original node objective
end
CombinedNode(index::Int64) = CombinedNode(index,Dict{Symbol,Any}(),Dict{JuMP.VariableRef,JuMP.VariableRef}(),
Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}(),Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}(),zero(JuMP.GenericAffExpr{Float64, JuMP.AbstractVariableRef}))

#Combined Info
mutable struct CombinedInfo
    nodes::Vector{CombinedNode}
    linkconstraints::Vector{ConstraintRef}
    NLlinkconstraints::Vector{ConstraintRef}
end
CombinedInfo() = CombinedInfo(CombinedNode[],ConstraintRef[],ConstraintRef[])

#A JuMP model created from a combined OptiGraph
function CombinedModel()
    m = JuMP.Model()
    m.ext[:CombinedInfo] = CombinedInfo()
    return m
end

is_combined_model(m::JuMP.Model) = haskey(m.ext,:CombinedInfo) ? true : false  #check if the model is a graph model
assert_is_combined_model(m::JuMP.Model) = @assert is_combined_model(m)
getcombinedinfo(m::JuMP.Model) = haskey(m.ext, :CombinedInfo) ? m.ext[:CombinedInfo] : error("Model is not a combined model")
getlinkconstraints(m::JuMP.Model) = is_combined_model(m) && getcombinedinfo(m).linkconstraints
getNLlinkconstraints(m::JuMP.Model) = is_combined_model(m) && getcombinedinfo(m).NLlinkconstraints
getnodes(m::JuMP.Model) = is_combined_model(m) && getcombinedinfo(m).nodes

#Create a new new node on an CombinedModel
function add_combined_node!(m::JuMP.Model)
    assert_is_combined_model(m)
    i = getnumnodes(m)
    agg_node = CombinedNode(i+1)
    push!(m.ext[:CombinedInfo].nodes,agg_node)
    return agg_node
end
getnumnodes(m::JuMP.Model) = length(getcombinedinfo(m).nodes)

JuMP.objective_function(node::CombinedNode) = node.objective
JuMP.num_variables(node::CombinedNode) = length(node.variablemap)
Base.getindex(node::CombinedNode,s::Symbol) = node.obj_dict[s]
getaggnodevariables(node::CombinedNode) = collect(keys(node.variablemap))
getaggnodeconstraints(node::CombinedNode) = collect(keys(node.constraintmap))

#############################################################################################
# CombinedMap
#############################################################################################
"""
    CombinedMap
    Mapping between variable and constraint reference of a OptiGraph to an Combined Model.
    The reference of the combined model can be obtained by indexing the map with the reference of the corresponding original modelnode.
"""
struct CombinedMap
    combined_model::JuMP.AbstractModel                             #An combined model (Could be another OptiGraph)
    varmap::Dict{JuMP.VariableRef,JuMP.VariableRef}                 #map variables in original modelgraph to combinedmodel
    conmap::Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}             #map constraints in original modelgraph to combinedmodel
    linkconstraintmap::Dict{LinkConstraint,JuMP.ConstraintRef}
end

function Base.getindex(reference_map::CombinedMap, vref::JuMP.VariableRef)  #reference_map[node_var] --> combinedd_copy_var
    return reference_map.varmap[vref]
end

function Base.getindex(reference_map::CombinedMap, cref::JuMP.ConstraintRef)
    return reference_map.conmap[cref]
end
Base.broadcastable(reference_map::CombinedMap) = Ref(reference_map)

function Base.setindex!(reference_map::CombinedMap, graph_cref::JuMP.ConstraintRef,node_cref::JuMP.ConstraintRef)
    reference_map.conmap[node_cref] = graph_cref
end

function Base.setindex!(reference_map::CombinedMap, graph_vref::JuMP.VariableRef,node_vref::JuMP.VariableRef)
    reference_map.varmap[node_vref] = graph_vref
end

CombinedMap(m::JuMP.AbstractModel) = CombinedMap(m,Dict{JuMP.VariableRef,JuMP.VariableRef}(),Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}(),Dict{LinkConstraintRef,JuMP.ConstraintRef}())

function Base.merge!(ref_map1::CombinedMap,ref_map2::CombinedMap)
    merge!(ref_map1.varmap,ref_map2.varmap)
    merge!(ref_map1.conmap,ref_map2.conmap)
end

#############################################################################################
# Combine Functions
#############################################################################################


function combine(modelgraph::JuMP.Model)
    combined_model = CombinedModel()
    reference_map = CombinedMap(combined_model)

    #COPY NODE MODELS INTO Combined MODEL
    for modelnode in getChildren(modelgraph)               #for each node in the model graph
        node_model = getmodel(modelnode)
        #Need to pass master reference so we use those variables instead of creating new ones
        node_ref_map = _add_to_combined_model!(combined_model,node_model,reference_map)  #updates combined_model and reference_map

        end
    end

    #OBJECTIVE FUNCTION
    if !(has_objective(modelgraph))
        _set_node_objectives!(modelgraph)  #set modelgraph objective function
    _set_node_objectives!(modelgraph,combined_model,reference_map,has_nonlinear_objective) #set combined_model objective function
    end

    if has_objective(modelgraph)
        agg_graph_obj = _copy_constraint_func(JuMP.objective_function(modelgraph),reference_map)
        JuMP.set_objective_function(combined_model,agg_graph_obj)
        JuMP.set_objective_sense(combined_model,JuMP.objective_sense(modelgraph))
    end

    #ADD LINK CONSTRAINTS
    for linkconstraint in all_linkconstraints(modelgraph)
        new_constraint = _copy_constraint(linkconstraint,reference_map)
        cref = JuMP.add_constraint(combined_model,new_constraint)
        reference_map.linkconstraintmap[linkconstraint] = cref
    end

    modelnode = OptiNode()
    set_model(modelnode,combined_model)
    return modelnode,reference_map
end

const aggregate = combine


function copy(node::OptiNode)
    node_model = getmodel(node)
    new_model = CombinedModel()
    reference_map = CombinedMap(new_model)
    node_ref_map = _add_to_combined_model!(new_model,node_model,reference_map)
    new_node = OptiNode()
    set_model(new_node,new_model)
    return new_node,reference_map
end

function combine(graph::OptiGraph,max_depth::Int64)  #0 means no subgraphs
    println("Creating Combined OptiGraph with a maximum subgraph depth of $max_depth")

    sg_dict = Dict()
    root_modelgraph = OptiGraph()
    reference_map = CombinedMap(root_modelgraph)  #old model graph => new modelgraph
    sg_dict[graph] = root_modelgraph

    #iterate through depth until we get to last level.  last level is the leaf subgraphs that get converted to nodes
    depth = 0
    parents = [graph]
    final_parents = [graph]
    while depth < max_depth  #maximum subgraph depth.  0 means no subgraphs
        subs_to_check = []
        for parent in parents
            new_parent = sg_dict[parent]
            subs = getsubgraphs(parent)
            for sub in subs
                new_subgraph = OptiGraph()
                add_subgraph!(new_parent,new_subgraph)
                sg_dict[sub] = new_subgraph
            end
            append!(subs_to_check,subs)
        end
        depth += 1
        append!(final_parents,parents)
        parents = subs_to_check
    end

    #ADD THE BOTTOM LEVEL NODES from the corresponding subgraphs
    for parent in parents
        name_idx = 1
        for leaf_subgraph in getsubgraphs(parent)
            combined_node,combine_ref_map = combine(leaf_subgraph) #creates new modelnode
            merge!(reference_map,combine_ref_map)
            new_parent = sg_dict[parent]
            add_node!(new_parent,combined_node)
            combined_node.label = "$name_idx'"
            name_idx += 1
        end
    end

    #Now add nodes and edges to the higher level graphs
    for graph in reverse(final_parents)  #reverse to start from the bottom
        name_idx = 1

        mnodes = getnodes(graph)
        ledges = getedges(graph)

        new_graph = sg_dict[graph]

        #Add copy modelnodes
        for node in mnodes
            new_node,ref_map = copy(node)
            merge!(reference_map,ref_map)
            add_node!(new_graph,new_node)
            new_node.label = "$name_idx'"
            name_idx += 1
        end

        #Add copy linkconstraints
        for linkedge in ledges
            for linkconstraint in getlinkconstraints(linkedge)
                new_con = _copy_constraint(linkconstraint,reference_map)
                JuMP.add_constraint(new_graph,new_con)
            end
        end
    end

    return root_modelgraph,reference_map
end

#Modify graph by combining subgraphs
#IDEA: Create new OptiGraph with subgraphs based on partition object.
#Group subgraphs together for solver interface
function _add_to_combined_model!(combined_model::JuMP.Model,node_model::JuMP.Model,aggregation_map::CombinedMap)

    global t_add_node += @elapsed begin
        agg_node = add_combined_node!(combined_model)
    end

    if JuMP.mode(node_model) == JuMP.DIRECT
        error("Cannot copy a node model in `DIRECT` mode. Use the `Model` ",
              "constructor instead of the `direct_model` constructor to be ",
              "able to combined into a new JuMP Model.")
    end

    reference_map = CombinedMap(combined_model)
    constraint_types = JuMP.list_of_constraint_types(node_model)

    #COPY VARIABLES
    global t_copy_vars += @elapsed begin
        for var in JuMP.all_variables(node_model)
            new_x = JuMP.@variable(combined_model)                      #create an anonymous variable
            reference_map[var] = new_x                                   #map variable reference to new reference
            var_name = JuMP.name(var)
            new_name = var_name
            JuMP.set_name(new_x,new_name)
            if JuMP.start_value(var) != nothing
                JuMP.set_start_value(new_x,JuMP.start_value(var))
            end
            agg_node.variablemap[new_x] = var
        end
    end

    global t_copy_constraints += @elapsed begin
        #COPY ALL OTHER CONSTRAINTS
        #Use JuMP and check if I have a ScalarConstraint or VectorConstraint and use the reference map to create new constraints
        for (func,set) in constraint_types
            constraint_refs = JuMP.all_constraints(node_model, func, set)
            for constraint_ref in constraint_refs
                constraint = JuMP.constraint_object(constraint_ref)
                new_constraint = _copy_constraint(constraint,reference_map)
                new_ref= JuMP.add_constraint(combined_model,new_constraint)
                agg_node.constraintmap[new_ref] = constraint_ref
                reference_map[constraint_ref] = new_ref
            end
        end
    end

    global t_copy_nl_constraints += @elapsed begin
        #COPY NONLINEAR CONSTRAINTS
        nlp_initialized = false
        if node_model.nlp_data !== nothing
            d = JuMP.NLPEvaluator(node_model)           #Get the NLP evaluator object.  Initialize the expression graph
            MOI.initialize(d,[:ExprGraph])
            nlp_initialized = true
            for i = 1:length(node_model.nlp_data.nlconstr)
                expr = MOI.constraint_expr(d,i)                         #this returns a julia expression
                _splice_nonlinear_variables!(expr,node_model,reference_map)        #splice the variables from var_map into the expression
                new_nl_constraint = JuMP.add_NL_constraint(combined_model,expr)      #raw expression input for non-linear constraint
                constraint_ref = JuMP.ConstraintRef(node_model,JuMP.NonlinearConstraintIndex(i),new_nl_constraint.shape)
                agg_node.nl_constraintmap[new_nl_constraint] = constraint_ref
                reference_map[constraint_ref] = new_nl_constraint
            end
        end
    end

    #TODO Get nonlinear object data to work.
    # COPY OBJECT DATA (JUMP CONTAINERS).
    # for (name, value) in JuMP.object_dictionary(node_model)
    #     #agg_node.obj_dict[name] = reference_map[value]
    #     if typeof(value) in [JuMP.VariableRef,JuMP.ConstraintRef,LinkVariableRef]
    #         agg_node.obj_dict[name] = getindex.(reference_map, value)
    #     end
    # end

    #OBJECTIVE FUNCTION (store expression on combinedd_nodes)
    if !(_has_nonlinear_obj(node_model))
        #AFFINE OR QUADTRATIC OBJECTIVE
        new_objective = _copy_objective(node_model,reference_map)
        agg_node.objective = new_objective
    else
        #NONLINEAR OBJECTIVE
        if !nlp_initialized
            d = JuMP.NLPEvaluator(node_model)           #Get the NLP evaluator object.  Initialize the expression graph
            MOI.initialize(d,[:ExprGraph])
        end
        new_obj = _copy_nl_objective(d,reference_map)
        agg_node.objective = new_obj
    end

    merge!(aggregation_map,reference_map)

    return reference_map
end

#Creata new set of nodes on a modelgraph
function _set_nodes(mg::OptiGraph,nodes::Vector{OptiNode})
    mg.modelnodes = nodes
    for (idx,node) in enumerate(mg.modelnodes)
        mg.node_idx_map[node] = idx
    end
    return nothing
end

#Create a new set of edges on a modelgraph
function _set_edges(mg::OptiGraph,edges::Vector{OptiEdge})
    mg.linkedges = edges
    link_idx = 0
    for (idx,linkedge) in enumerate(mg.linkedges)
        mg.edge_idx_map[linkedge] = idx
        mg.linkedge_map[linkedge.nodes] = linkedge
    end
end

#Set combined model objective to sum of OptiGraph node objectives
function _set_node_objectives!(modelgraph::OptiGraph,combined_model::JuMP.Model,reference_map::CombinedMap,has_nonlinear_objective::Bool)
    if has_nonlinear_objective
        graph_obj = :(0) #NOTE Strategy: Build up a Julia expression (expr) and then call JuMP.set_NL_objective(expr)
        for node in all_nodes(modelgraph)
            node_model = getmodel(node)
            JuMP.objective_sense(node_model) == MOI.MIN_SENSE ? sense = 1 : sense = -1
            d = JuMP.NLPEvaluator(node_model)
            MOI.initialize(d,[:ExprGraph])
            node_obj = MOI.objective_expr(d)
            _splice_nonlinear_variables!(node_obj,node_model,reference_map)  #_splice_nonlinear_variables!(node_obj,var_maps[node])
            node_obj = Expr(:call,:*,:($sense),node_obj)
            graph_obj = Expr(:call,:+,graph_obj,node_obj)  #update graph objective
        end
        JuMP.set_NL_objective(combined_model, MOI.MIN_SENSE, graph_obj)
    else
        #TODO: Fix issue with setting maximize
        graph_obj = sum(JuMP.objective_function(agg_node) for agg_node in getnodes(combined_model))    #NOTE: All of the node objectives are converted to Minimize (MOI.OptimizationSense(0))
        JuMP.set_objective(combined_model,MOI.MIN_SENSE,graph_obj)
    end
end

function _set_node_objectives!(modelgraph::OptiGraph)
    #check for quadratic objectives
    if any(isa.(objective_function.(all_nodes(modelgraph)),Ref(GenericQuadExpr)))
        graph_obj = zero(JuMP.GenericQuadExpr{Float64, JuMP.VariableRef})
    else
        graph_obj = zero(JuMP.GenericAffExpr{Float64, JuMP.VariableRef})
    end

    #  #testing changing this to quadratic expression
    for node in all_nodes(modelgraph)
        sense = JuMP.objective_sense(node)
        s = sense == MOI.MAX_SENSE ? -1.0 : 1.0
        JuMP.add_to_expression!(graph_obj,s,JuMP.objective_function(node))
    end

    JuMP.set_objective(modelgraph,MOI.MIN_SENSE,graph_obj)
end
